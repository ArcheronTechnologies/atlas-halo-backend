from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Response
from sqlalchemy.orm import Session
from ..db.memory import db
from ..db.mongo import get_mongo_db
from ..db.session import get_session
from ..models.ingestion import EmailBatchRequest, WebDataRequest
from ..models.pricing import PriceHistoryBatch
from ..search.indexer import index_document
from ..integrations.microsoft365 import create_microsoft365_integration
from ..core.config import settings
import logging
import os
from typing import Dict, Any, List
from datetime import datetime, timezone
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/email-batch")
async def email_batch(body: EmailBatchRequest):
    mongo = get_mongo_db()
    import re
    part_re = re.compile(r"\b[A-Z0-9]{2,}[-A-Z0-9]{2,}\b")
    emails = []
    for e in body.emails:
        doc = e.model_dump()
        text = (doc.get("subject") or "") + "\n" + (doc.get("body") or "")
        parts = list({p for p in part_re.findall(text) if len(p) >= 5})
        doc.setdefault("classification", "rfq" if "RFQ" in text.upper() else "other")
        doc.setdefault("extractedData", {})
        doc["extractedData"]["components"] = parts
        emails.append(doc)
    if mongo:
        coll = mongo.get_collection("emails")
        await coll.insert_many(emails)
        return {"received": len(emails), "stored": "mongodb"}
    # Fallback to in-memory store
    count = db.add_emails(emails)
    return {"received": count, "stored": "memory"}


@router.post("/web-data")
async def web_data(body: WebDataRequest):
    doc = body.model_dump()
    mongo = get_mongo_db()
    if mongo:
        coll = mongo.get_collection("web_data")
        await coll.insert_one(doc)
    else:
        db.add_web_data(doc)
    # Best-effort index in Elasticsearch
    index_document({
        "title": doc.get("url"),
        "content": str(doc.get("extractedData")),
        "url": doc.get("url"),
        "domain": doc.get("url"),
    })
    return {"stored": True, "url": doc["url"]}


@router.get("/microsoft365/webhook")
async def microsoft365_webhook_validation(validationToken: str | None = None):
    # Echo the validation token for subscription validation
    if validationToken:
        return Response(content=validationToken, media_type="text/plain")
    return {"status": "ok"}


@router.post("/microsoft365/webhook")
async def microsoft365_webhook_notifications(payload: Dict[str, Any]):
    mongo = get_mongo_db()
    if mongo:
        try:
            coll = mongo.get_collection("graph_notifications")
            await coll.insert_one({"payload": payload, "receivedAt": datetime.now(timezone.utc)})
        except Exception:
            pass
    return {"received": True}


@router.post("/microsoft365/subscriptions")
async def microsoft365_create_subscription(body: Dict[str, Any]):
    # Placeholder: create subscription via Graph API
    return {"created": True, "resource": body.get("resource")}


@router.post("/microsoft365/sync-emails")
async def sync_microsoft365_emails(
    background_tasks: BackgroundTasks,
    hours_back: int = 24,
    mailbox: str = "me",
    session: Session = Depends(get_session)
):
    """
    Sync emails from Microsoft 365 with intelligent processing
    """
    try:
        # Get Microsoft 365 configuration
        config = {
            "tenant_id": os.getenv("M365_TENANT_ID"),
            "client_id": os.getenv("M365_CLIENT_ID"),
            "client_secret": os.getenv("M365_CLIENT_SECRET")
        }
        
        if not all(config.values()):
            raise HTTPException(
                status_code=400,
                detail="Microsoft 365 configuration not complete. Please set M365_TENANT_ID, M365_CLIENT_ID, and M365_CLIENT_SECRET environment variables."
            )
        
        # Create integration instance
        m365 = create_microsoft365_integration(config)
        
        # Start background email processing
        background_tasks.add_task(
            _process_microsoft365_emails,
            m365, mailbox, hours_back, session
        )
        
        return {
            "status": "started",
            "message": f"Email sync initiated for last {hours_back} hours",
            "mailbox": mailbox
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating Microsoft 365 email sync: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate email sync"
        )


@router.post("/microsoft365/sync-teams")
async def sync_microsoft365_teams(
    background_tasks: BackgroundTasks,
    hours_back: int = 24,
    team_id: str = None,
    session: Session = Depends(get_session)
):
    """
    Sync Teams messages from Microsoft 365
    """
    try:
        # Get Microsoft 365 configuration
        config = {
            "tenant_id": os.getenv("M365_TENANT_ID"),
            "client_id": os.getenv("M365_CLIENT_ID"),
            "client_secret": os.getenv("M365_CLIENT_SECRET")
        }
        
        if not all(config.values()):
            raise HTTPException(
                status_code=400,
                detail="Microsoft 365 configuration not complete"
            )
        
        # Create integration instance
        m365 = create_microsoft365_integration(config)
        
        # Start background Teams processing
        background_tasks.add_task(
            _process_microsoft365_teams,
            m365, team_id, hours_back, session
        )
        
        return {
            "status": "started",
            "message": f"Teams sync initiated for last {hours_back} hours",
            "team_id": team_id or "all_teams"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating Microsoft 365 Teams sync: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate Teams sync"
        )


@router.get("/microsoft365/status")
async def get_sync_status():
    """
    Get status of Microsoft 365 integration
    """
    config_status = {
        "tenant_id": bool(os.getenv("M365_TENANT_ID")),
        "client_id": bool(os.getenv("M365_CLIENT_ID")),
        "client_secret": bool(os.getenv("M365_CLIENT_SECRET"))
    }
    
    is_configured = all(config_status.values())
    
    return {
        "configured": is_configured,
        "config_status": config_status,
        "last_sync": None,  # Would be stored in database
        "status": "ready" if is_configured else "needs_configuration"
    }


# Background task functions
async def _process_microsoft365_emails(m365, mailbox: str, hours_back: int, session: Session):
    """Background task to process Microsoft 365 emails"""
    try:
        logger.info(f"Starting Microsoft 365 email processing for {mailbox}")
        
        processed_count = 0
        error_count = 0
        
        async for email in m365.process_mailbox_emails(mailbox=mailbox, hours_back=hours_back):
            try:
                # Store in MongoDB
                mongo = get_mongo_db()
                if mongo:
                    coll = mongo.get_collection("processed_emails")
                    email_dict = {
                        "messageId": email.id,
                        "threadId": email.thread_id,
                        "sender": email.sender,
                        "recipients": email.recipients,
                        "subject": email.subject,
                        "bodyText": email.body_text,
                        "bodyHtml": email.body_html,
                        "receivedAt": email.received_at,
                        "attachments": email.attachments,
                        "classification": email.classification,
                        "extractedData": email.extracted_data,
                        "processedAt": datetime.now(timezone.utc)
                    }
                    await coll.insert_one(email_dict)
                
                # Index in Elasticsearch for search
                index_document({
                    "title": email.subject,
                    "content": email.body_text,
                    "sender": email.sender.get("email", ""),
                    "type": "email",
                    "classification": email.classification.get("type") if email.classification else "unknown"
                })
                
                # If this is an RFQ, create RFQ record
                if email.classification and email.classification.get("type") == "rfq":
                    await _create_rfq_from_email(email, session)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing email {email.id}: {e}")
                error_count += 1
                continue
        
        logger.info(f"Microsoft 365 email processing completed. Processed: {processed_count}, Errors: {error_count}")
        
    except Exception as e:
        logger.error(f"Error in Microsoft 365 email processing task: {e}")


async def _process_microsoft365_teams(m365, team_id: str, hours_back: int, session: Session):
    """Background task to process Microsoft 365 Teams messages"""
    try:
        logger.info(f"Starting Microsoft 365 Teams processing")
        
        processed_count = 0
        error_count = 0
        
        async for message in m365.process_teams_messages(team_id=team_id, hours_back=hours_back):
            try:
                # Store in MongoDB
                mongo = get_mongo_db()
                if mongo:
                    coll = mongo.get_collection("teams_messages")
                    message_dict = {
                        "messageId": message.id,
                        "chatId": message.chat_id,
                        "channelId": message.channel_id,
                        "sender": message.sender,
                        "content": message.content,
                        "messageType": message.message_type,
                        "createdAt": message.created_at,
                        "mentions": message.mentions,
                        "attachments": message.attachments,
                        "sentiment": getattr(message, 'sentiment', None),
                        "entities": getattr(message, 'entities', None),
                        "processedAt": datetime.now(timezone.utc)
                    }
                    await coll.insert_one(message_dict)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing Teams message {message.id}: {e}")
                error_count += 1
                continue
        
        logger.info(f"Microsoft 365 Teams processing completed. Processed: {processed_count}, Errors: {error_count}")
        
    except Exception as e:
        logger.error(f"Error in Microsoft 365 Teams processing task: {e}")


async def _create_rfq_from_email(email, session: Session):
    """Create RFQ record from classified email"""
    try:
        from ..repositories.rfqs_repo import RFQsRepository
        from ..repositories.companies_repo import CompaniesRepository
        import uuid
        
        # Try to find or create customer company
        companies_repo = CompaniesRepository(session)
        sender_email = email.sender.get("email", "")
        
        # Simple domain-based company matching
        domain = sender_email.split("@")[-1] if "@" in sender_email else "unknown.com"
        
        # Check if we have a company with this domain (simplified)
        # In production, this would be more sophisticated
        customer = companies_repo.create({
            "name": email.sender.get("name", domain),
            "type": "customer",
            "website": f"https://{domain}",
        })
        
        # Create RFQ
        rfqs_repo = RFQsRepository(session)
        rfq_data = {
            "customerId": customer.id,
            "rfqNumber": f"EMAIL-{email.id[:8]}",
            "source": "email",
            "sourceReference": email.id,
            "items": []
        }
        
        # Add items if extracted from email
        if email.extracted_data and email.extracted_data.get("components"):
            for i, component in enumerate(email.extracted_data["components"][:5]):  # Max 5 items
                quantity = 1
                if i < len(email.extracted_data.get("quantities", [])):
                    quantity = email.extracted_data["quantities"][i]
                
                target_price = None
                if i < len(email.extracted_data.get("pricing", [])):
                    target_price = email.extracted_data["pricing"][i]
                
                rfq_data["items"].append({
                    "customerPartNumber": component,
                    "quantity": quantity,
                    "targetPrice": target_price
                })
        
        rfq = rfqs_repo.create(rfq_data)
        logger.info(f"Created RFQ {rfq.id} from email {email.id}")
        
    except Exception as e:
        logger.error(f"Error creating RFQ from email {email.id}: {e}")


@router.post("/pricing")
async def pricing_batch(body: PriceHistoryBatch, session: Session = Depends(get_session)):
    now = datetime.now(timezone.utc)
    from ..db.models import PriceHistory
    import uuid as _uuid
    recs = [
        PriceHistory(
            id=str(_uuid.uuid4()),
            component_id=p.componentId,
            supplier_id=p.supplierId,
            quantity_break=p.quantityBreak,
            unit_price=p.unitPrice,
            currency=p.currency or "USD",
            source_type=p.sourceType,
            source_reference=p.sourceReference,
            created_at=now,
        )
        for p in body.prices
    ]
    if recs:
        session.add_all(recs)
        session.commit()
    return {"ingested": len(recs)}


# Additional ingestion endpoints for Phase 2

class EmailIngestionRequest(BaseModel):
    sender: str
    recipients: List[str]
    subject: str
    body: str
    attachments: List[Dict[str, Any]] = []
    message_id: str = None
    thread_id: str = None


class BOMIngestionRequest(BaseModel):
    bom_name: str
    components: List[Dict[str, Any]]
    source: str = "manual"
    version: str = "1.0"


class ERPDataRequest(BaseModel):
    data_type: str
    records: List[Dict[str, Any]]
    source_system: str
    timestamp: datetime = None


class TeamsMessageRequest(BaseModel):
    message_id: str
    channel_id: str
    sender: str
    content: str
    timestamp: datetime
    mentions: List[str] = []


class WebIntelligenceRequest(BaseModel):
    url: str
    content: str
    extracted_data: Dict[str, Any]
    source_type: str = "web_crawl"


@router.post("/email")
async def ingest_email(
    request: EmailIngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest individual email for processing and analysis
    """
    try:
        email_data = {
            "messageId": request.message_id or f"email_{datetime.now().timestamp()}",
            "threadId": request.thread_id,
            "sender": request.sender,
            "recipients": request.recipients,
            "subject": request.subject,
            "body": request.body,
            "attachments": request.attachments,
            "receivedAt": datetime.now(timezone.utc),
            "processedAt": datetime.now(timezone.utc),
            "source": "api_ingestion"
        }
        
        # Store in MongoDB
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("ingested_emails")
            await coll.insert_one(email_data)
        else:
            # Fallback to in-memory storage
            db.add_emails([email_data])
        
        # Add AI processing task
        background_tasks.add_task(_process_email_ai, email_data)
        
        return {
            "status": "success",
            "message_id": email_data["messageId"],
            "processed": True
        }
        
    except Exception as e:
        logger.error(f"Error ingesting email: {e}")
        raise HTTPException(status_code=500, detail="Email ingestion failed")


@router.post("/bom")
async def ingest_bom(
    request: BOMIngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest Bill of Materials for analysis and optimization
    """
    try:
        bom_data = {
            "bomId": f"bom_{datetime.now().timestamp()}",
            "name": request.bom_name,
            "components": request.components,
            "source": request.source,
            "version": request.version,
            "ingestedAt": datetime.now(timezone.utc),
            "totalComponents": len(request.components)
        }
        
        # Store in MongoDB
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("ingested_boms")
            await coll.insert_one(bom_data)
        
        # Add AI processing task for BOM optimization
        background_tasks.add_task(_process_bom_ai, bom_data)
        
        return {
            "status": "success",
            "bom_id": bom_data["bomId"],
            "components_processed": len(request.components)
        }
        
    except Exception as e:
        logger.error(f"Error ingesting BOM: {e}")
        raise HTTPException(status_code=500, detail="BOM ingestion failed")


@router.post("/erp/visma")
async def ingest_visma_data(
    request: ERPDataRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest data from Visma ERP system
    """
    try:
        erp_data = {
            "dataType": request.data_type,
            "records": request.records,
            "sourceSystem": "visma",
            "ingestedAt": datetime.now(timezone.utc),
            "recordCount": len(request.records),
            "timestamp": request.timestamp or datetime.now(timezone.utc)
        }
        
        # Store in MongoDB
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("erp_data")
            await coll.insert_one(erp_data)
        
        # Add processing task
        background_tasks.add_task(_process_erp_data, erp_data)
        
        return {
            "status": "success",
            "data_type": request.data_type,
            "records_processed": len(request.records)
        }
        
    except Exception as e:
        logger.error(f"Error ingesting Visma ERP data: {e}")
        raise HTTPException(status_code=500, detail="Visma ERP ingestion failed")


@router.post("/microsoft365/teams")
async def ingest_teams_message(
    request: TeamsMessageRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest Microsoft Teams message for analysis
    """
    try:
        teams_data = {
            "messageId": request.message_id,
            "channelId": request.channel_id,
            "sender": request.sender,
            "content": request.content,
            "timestamp": request.timestamp,
            "mentions": request.mentions,
            "ingestedAt": datetime.now(timezone.utc),
            "source": "teams_api"
        }
        
        # Store in MongoDB
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("teams_messages")
            await coll.insert_one(teams_data)
        
        # Add AI processing task
        background_tasks.add_task(_process_teams_message_ai, teams_data)
        
        return {
            "status": "success",
            "message_id": request.message_id,
            "processed": True
        }
        
    except Exception as e:
        logger.error(f"Error ingesting Teams message: {e}")
        raise HTTPException(status_code=500, detail="Teams message ingestion failed")


@router.post("/web-intelligence")
async def ingest_web_intelligence(
    request: WebIntelligenceRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest web intelligence data for market analysis
    """
    try:
        web_data = {
            "url": request.url,
            "content": request.content,
            "extractedData": request.extracted_data,
            "sourceType": request.source_type,
            "ingestedAt": datetime.now(timezone.utc),
            "processed": False
        }
        
        # Store in MongoDB
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("web_intelligence")
            await coll.insert_one(web_data)
        
        # Index for search
        index_document({
            "title": request.url,
            "content": request.content,
            "url": request.url,
            "type": "web_intelligence",
            "extracted_data": str(request.extracted_data)
        })
        
        # Add AI processing task
        background_tasks.add_task(_process_web_intelligence_ai, web_data)
        
        return {
            "status": "success",
            "url": request.url,
            "processed": True
        }
        
    except Exception as e:
        logger.error(f"Error ingesting web intelligence: {e}")
        raise HTTPException(status_code=500, detail="Web intelligence ingestion failed")


# Background processing tasks

async def _process_email_ai(email_data: Dict[str, Any]):
    """Process email using AI capabilities"""
    try:
        from ..ai.capabilities import extract_components, classify_intent
        
        # Extract components from email
        text = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
        components_result = await extract_components(text)
        
        # Classify intent
        intent_result = await classify_intent(text, email_data.get('subject', ''))
        
        # Update email with AI results
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("ingested_emails")
            await coll.update_one(
                {"messageId": email_data["messageId"]},
                {
                    "$set": {
                        "aiProcessing": {
                            "components": components_result.data if components_result.success else [],
                            "intent": intent_result.data if intent_result.success else {},
                            "processedAt": datetime.now(timezone.utc)
                        }
                    }
                }
            )
        
        logger.info(f"AI processing completed for email {email_data['messageId']}")
        
    except Exception as e:
        logger.error(f"Error in AI processing for email: {e}")


async def _process_bom_ai(bom_data: Dict[str, Any]):
    """Process BOM using AI capabilities"""
    try:
        from ..ai.capabilities import get_recommendations
        
        # Get recommendations for each component
        recommendations = []
        for component in bom_data.get("components", []):
            result = await get_recommendations(component.get("part_number", ""))
            if result.success:
                recommendations.append({
                    "component": component,
                    "recommendations": result.data
                })
        
        # Update BOM with AI results
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("ingested_boms")
            await coll.update_one(
                {"bomId": bom_data["bomId"]},
                {
                    "$set": {
                        "aiProcessing": {
                            "recommendations": recommendations,
                            "processedAt": datetime.now(timezone.utc)
                        }
                    }
                }
            )
        
        logger.info(f"AI processing completed for BOM {bom_data['bomId']}")
        
    except Exception as e:
        logger.error(f"Error in AI processing for BOM: {e}")


async def _process_erp_data(erp_data: Dict[str, Any]):
    """Process ERP data"""
    try:
        # Simple processing - in a real system this would involve
        # data validation, transformation, and integration
        logger.info(f"Processing ERP data: {erp_data['dataType']}")
        
        # Update processing status
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("erp_data")
            await coll.update_one(
                {"_id": erp_data.get("_id")},
                {
                    "$set": {
                        "processed": True,
                        "processedAt": datetime.now(timezone.utc)
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"Error processing ERP data: {e}")


async def _process_teams_message_ai(teams_data: Dict[str, Any]):
    """Process Teams message using AI"""
    try:
        from ..ai.capabilities import classify_intent
        
        # Analyze sentiment/intent of Teams message
        intent_result = await classify_intent(teams_data.get("content", ""))
        
        # Update message with AI results
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("teams_messages")
            await coll.update_one(
                {"messageId": teams_data["messageId"]},
                {
                    "$set": {
                        "aiProcessing": {
                            "intent": intent_result.data if intent_result.success else {},
                            "processedAt": datetime.now(timezone.utc)
                        }
                    }
                }
            )
        
        logger.info(f"AI processing completed for Teams message {teams_data['messageId']}")
        
    except Exception as e:
        logger.error(f"Error in AI processing for Teams message: {e}")


async def _process_web_intelligence_ai(web_data: Dict[str, Any]):
    """Process web intelligence using AI"""
    try:
        from ..ai.capabilities import extract_components
        
        # Extract components and market information
        components_result = await extract_components(web_data.get("content", ""))
        
        # Update web data with AI results
        mongo = get_mongo_db()
        if mongo:
            coll = mongo.get_collection("web_intelligence")
            await coll.update_one(
                {"url": web_data["url"]},
                {
                    "$set": {
                        "aiProcessing": {
                            "components": components_result.data if components_result.success else [],
                            "processedAt": datetime.now(timezone.utc)
                        },
                        "processed": True
                    }
                }
            )
        
        logger.info(f"AI processing completed for web intelligence {web_data['url']}")
        
    except Exception as e:
        logger.error(f"Error in AI processing for web intelligence: {e}")
