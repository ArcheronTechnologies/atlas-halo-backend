#!/usr/bin/env python3
"""
Enhanced Data Sources for Atlas AI
Integrates additional Swedish crime and safety data sources beyond Polisen.se
"""

import asyncio
import aiohttp
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import feedparser
import re
from urllib.parse import urljoin, urlparse
import csv
import io

try:
    from .data_quality_validator import EnhancedDataQualityValidator
except ImportError:
    from data_quality_validator import EnhancedDataQualityValidator

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    url: str
    source_type: str  # 'api', 'rss', 'csv', 'json'
    update_frequency_hours: int
    reliability_score: float
    data_format: str
    active: bool = True
    last_collected: Optional[datetime] = None
    

class BRAStatisticsConnector:
    """
    Connector for BRÅ (Swedish National Council for Crime Prevention) data
    Provides crime statistics and trend data
    """
    
    BASE_URL = "https://www.bra.se"
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session or aiohttp.ClientSession()
        self.cache_dir = Path("data_lake/cache/bra")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def fetch_crime_statistics(self, year: int = None) -> List[Dict[str, Any]]:
        """Fetch crime statistics from BRÅ"""
        
        year = year or datetime.now().year - 1  # Previous year data
        
        try:
            logger.info(f"Fetching BRÅ crime statistics for {year}")
            
            # BRÅ provides various statistical endpoints
            # This would need to be adapted based on their actual API structure
            stats_urls = [
                "https://www.bra.se/statistik/kriminalstatistik.html",
                "https://www.bra.se/statistik/statistiska-undersokningar.html"
            ]
            
            statistics = []
            
            for url in stats_urls:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            # Parse statistical content
                            parsed_stats = self._parse_bra_statistics(content, url)
                            statistics.extend(parsed_stats)
                        else:
                            logger.warning(f"Failed to fetch BRÅ data from {url}: {response.status}")
                
                except Exception as e:
                    logger.error(f"Error fetching BRÅ statistics from {url}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(statistics)} BRÅ statistical records")
            return statistics
            
        except Exception as e:
            logger.error(f"Error in BRÅ statistics collection: {e}")
            return []
    
    def _parse_bra_statistics(self, content: str, source_url: str) -> List[Dict[str, Any]]:
        """Parse BRÅ statistical content"""
        
        # This would need proper parsing based on BRÅ's actual data format
        # For now, return structured placeholder data
        
        statistics = []
        
        # Look for statistical tables or data in the content
        # This is a simplified example - real implementation would use proper HTML parsing
        
        crime_keywords = ['brott', 'våld', 'stöld', 'rån', 'narkotika']
        
        for keyword in crime_keywords:
            if keyword in content.lower():
                stat_record = {
                    'id': f"bra_stat_{keyword}_{datetime.now().strftime('%Y%m%d')}",
                    'crime_type': keyword,
                    'source': 'bra_statistics',
                    'source_url': source_url,
                    'year': datetime.now().year - 1,
                    'data_type': 'statistical_trend',
                    'collected_at': datetime.utcnow().isoformat(),
                    'raw_content': content[:500]  # Sample of content
                }
                statistics.append(stat_record)
        
        return statistics


class MunicipalDataConnector:
    """
    Enhanced connector for Swedish municipal open data
    Includes Stockholm, Malmö, Göteborg and other major cities
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session or aiohttp.ClientSession()
        
        # Municipal data sources
        self.municipal_sources = {
            'stockholm': {
                'name': 'Stockholm Stad',
                'base_url': 'https://opendata.stockholm.se/',
                'endpoints': {
                    'incidents': 'datastore/odata3.0/df8c1a84-4e39-4bb3-9a84-2b7e5b0c97be',
                    'safety_reports': 'datastore/odata3.0/7e2b7b84-1234-4567-8901-234567890123'
                }
            },
            'malmo': {
                'name': 'Malmö Stad', 
                'base_url': 'https://ckan-malmo.dataplatform.se/',
                'endpoints': {
                    'safety_data': 'api/3/action/package_search?q=säkerhet',
                    'incident_reports': 'api/3/action/package_search?q=händelser'
                }
            },
            'goteborg': {
                'name': 'Göteborgs Stad',
                'base_url': 'https://goteborg.se/wps/portal/start/kommun-o-politik/kommunfakta/oppna-data/',
                'endpoints': {
                    'public_safety': 'api/public-safety'
                }
            }
        }
    
    async def fetch_stockholm_data(self) -> List[Dict[str, Any]]:
        """Fetch safety-related data from Stockholm open data"""
        
        try:
            logger.info("Fetching Stockholm municipal data")
            
            incidents = []
            stockholm_config = self.municipal_sources['stockholm']
            
            for endpoint_name, endpoint_path in stockholm_config['endpoints'].items():
                url = urljoin(stockholm_config['base_url'], endpoint_path)
                
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process Stockholm OData format
                            processed_incidents = self._process_stockholm_data(data, endpoint_name)
                            incidents.extend(processed_incidents)
                            
                        else:
                            logger.warning(f"Stockholm endpoint {endpoint_name} returned {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching Stockholm {endpoint_name}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(incidents)} Stockholm municipal records")
            return incidents
            
        except Exception as e:
            logger.error(f"Error in Stockholm data collection: {e}")
            return []
    
    async def fetch_malmo_data(self) -> List[Dict[str, Any]]:
        """Fetch safety-related data from Malmö CKAN portal"""
        
        try:
            logger.info("Fetching Malmö municipal data")
            
            incidents = []
            malmo_config = self.municipal_sources['malmo']
            
            for endpoint_name, endpoint_path in malmo_config['endpoints'].items():
                url = urljoin(malmo_config['base_url'], endpoint_path)
                
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process CKAN format
                            processed_incidents = self._process_ckan_data(data, 'malmo', endpoint_name)
                            incidents.extend(processed_incidents)
                            
                        else:
                            logger.warning(f"Malmö endpoint {endpoint_name} returned {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching Malmö {endpoint_name}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(incidents)} Malmö municipal records")
            return incidents
            
        except Exception as e:
            logger.error(f"Error in Malmö data collection: {e}")
            return []
    
    def _process_stockholm_data(self, data: Dict[str, Any], endpoint_name: str) -> List[Dict[str, Any]]:
        """Process Stockholm OData format"""
        
        processed = []
        
        # Stockholm uses OData format
        if 'value' in data:
            for record in data['value']:
                processed_record = {
                    'id': f"stockholm_{endpoint_name}_{record.get('id', 'unknown')}",
                    'title': record.get('title', record.get('namn', 'Stockholm incident')),
                    'description': record.get('description', record.get('beskrivning', '')),
                    'datetime': self._parse_stockholm_datetime(record),
                    'location': self._extract_stockholm_location(record),
                    'source': 'stockholm_opendata',
                    'source_type': endpoint_name,
                    'crime_type': self._classify_stockholm_incident(record),
                    'raw_data': record,
                    'processed_at': datetime.utcnow().isoformat()
                }
                processed.append(processed_record)
        
        return processed
    
    def _process_ckan_data(self, data: Dict[str, Any], city: str, endpoint_name: str) -> List[Dict[str, Any]]:
        """Process CKAN format data"""
        
        processed = []
        
        if 'result' in data and 'results' in data['result']:
            for package in data['result']['results']:
                # Extract resources from CKAN packages
                if 'resources' in package:
                    for resource in package['resources']:
                        processed_record = {
                            'id': f"{city}_{endpoint_name}_{resource.get('id', 'unknown')}",
                            'title': package.get('title', f"{city.title()} safety data"),
                            'description': package.get('notes', resource.get('description', '')),
                            'datetime': self._parse_ckan_datetime(package, resource),
                            'location': {'city': city, 'region': city},
                            'source': f'{city}_opendata',
                            'source_type': endpoint_name,
                            'data_url': resource.get('url', ''),
                            'format': resource.get('format', ''),
                            'raw_data': {'package': package, 'resource': resource},
                            'processed_at': datetime.utcnow().isoformat()
                        }
                        processed.append(processed_record)
        
        return processed
    
    def _parse_stockholm_datetime(self, record: Dict[str, Any]) -> str:
        """Parse Stockholm datetime fields"""
        
        datetime_fields = ['datum', 'tidpunkt', 'created_at', 'updated_at']
        
        for field in datetime_fields:
            if field in record and record[field]:
                try:
                    # Handle various Stockholm datetime formats
                    dt_str = str(record[field])
                    if 'T' in dt_str:
                        return dt_str
                    else:
                        # Assume date only, add time
                        return f"{dt_str}T00:00:00"
                except:
                    continue
        
        return datetime.utcnow().isoformat()
    
    def _parse_ckan_datetime(self, package: Dict[str, Any], resource: Dict[str, Any]) -> str:
        """Parse CKAN datetime fields"""
        
        # Check resource first, then package
        for source in [resource, package]:
            for field in ['last_modified', 'created', 'metadata_created']:
                if field in source and source[field]:
                    return source[field]
        
        return datetime.utcnow().isoformat()
    
    def _extract_stockholm_location(self, record: Dict[str, Any]) -> Dict[str, str]:
        """Extract location from Stockholm record"""
        
        location = {'city': 'Stockholm', 'region': 'Stockholm'}
        
        # Look for various location fields
        location_fields = ['plats', 'adress', 'område', 'stadsdel']
        
        for field in location_fields:
            if field in record and record[field]:
                location['address'] = str(record[field])
                break
        
        return location
    
    def _classify_stockholm_incident(self, record: Dict[str, Any]) -> str:
        """Classify Stockholm incident type"""
        
        text = ' '.join(str(v) for v in record.values()).lower()
        
        if any(word in text for word in ['säkerhet', 'trygghet', 'brott']):
            return 'safety_incident'
        elif any(word in text for word in ['trafik', 'olycka']):
            return 'traffic_incident'
        else:
            return 'other'


class SocialMediaConnector:
    """
    Connector for social media safety reports and citizen reports
    Note: This would require proper API access and content moderation
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session or aiohttp.ClientSession()
        self.validator = EnhancedDataQualityValidator()
    
    async def fetch_twitter_safety_reports(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch safety-related reports from Twitter/X
        Note: Requires Twitter API access - placeholder implementation
        """
        
        keywords = keywords or ['säkerhet', 'brott', 'polis', 'incident', 'Stockholm', 'Malmö']
        
        logger.info("Twitter connector requires API keys - returning placeholder data")
        
        # Placeholder for Twitter integration
        # In production, this would use Twitter API v2
        
        return []
    
    async def fetch_community_reports(self) -> List[Dict[str, Any]]:
        """
        Fetch community safety reports from various platforms
        Placeholder for community reporting integration
        """
        
        logger.info("Community reporting connector - placeholder implementation")
        
        # This could integrate with:
        # - FixMyStreet equivalents in Sweden
        # - Municipal reporting apps
        # - Community safety platforms
        
        return []


class EnhancedDataSourceManager:
    """
    Manager for all enhanced data sources
    Orchestrates collection from multiple sources with quality validation
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session or aiohttp.ClientSession()
        self.validator = EnhancedDataQualityValidator()
        
        # Initialize connectors
        self.bra_connector = BRAStatisticsConnector(self.session)
        self.municipal_connector = MunicipalDataConnector(self.session)
        self.social_connector = SocialMediaConnector(self.session)
        
        # Track collection metrics
        self.collection_stats = {
            'sources_attempted': [],
            'sources_successful': [],
            'total_records': 0,
            'quality_scores': []
        }
    
    async def collect_all_enhanced_sources(self, hours_back: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data from all enhanced sources"""
        
        logger.info("Starting enhanced data source collection")
        
        all_data = {
            'bra_statistics': [],
            'stockholm_municipal': [],
            'malmo_municipal': [],
            'social_media': []
        }
        
        # Collect from BRÅ statistics
        try:
            logger.info("Collecting BRÅ statistics...")
            bra_data = await self.bra_connector.fetch_crime_statistics()
            all_data['bra_statistics'] = bra_data
            self.collection_stats['sources_attempted'].append('bra')
            if bra_data:
                self.collection_stats['sources_successful'].append('bra')
        except Exception as e:
            logger.error(f"BRÅ collection failed: {e}")
        
        # Collect from Stockholm
        try:
            logger.info("Collecting Stockholm municipal data...")
            stockholm_data = await self.municipal_connector.fetch_stockholm_data()
            all_data['stockholm_municipal'] = stockholm_data
            self.collection_stats['sources_attempted'].append('stockholm')
            if stockholm_data:
                self.collection_stats['sources_successful'].append('stockholm')
        except Exception as e:
            logger.error(f"Stockholm collection failed: {e}")
        
        # Collect from Malmö
        try:
            logger.info("Collecting Malmö municipal data...")
            malmo_data = await self.municipal_connector.fetch_malmo_data()
            all_data['malmo_municipal'] = malmo_data
            self.collection_stats['sources_attempted'].append('malmo')
            if malmo_data:
                self.collection_stats['sources_successful'].append('malmo')
        except Exception as e:
            logger.error(f"Malmö collection failed: {e}")
        
        # Quality validation for all collected data
        for source_name, data_list in all_data.items():
            if data_list:
                logger.info(f"Validating {len(data_list)} records from {source_name}")
                
                for record in data_list:
                    assessment = self.validator.validate_incident(record)
                    record['quality_assessment'] = {
                        'score': assessment.score,
                        'level': assessment.level.name,
                        'confidence': assessment.confidence,
                        'issues': assessment.issues,
                        'duplicate_risk': assessment.duplicate_risk
                    }
                    self.collection_stats['quality_scores'].append(assessment.score)
        
        # Update collection statistics
        total_records = sum(len(data_list) for data_list in all_data.values())
        self.collection_stats['total_records'] = total_records
        
        logger.info(f"Enhanced data collection completed: {total_records} total records")
        logger.info(f"Sources successful: {len(self.collection_stats['sources_successful'])}/{len(self.collection_stats['sources_attempted'])}")
        
        return all_data
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of last collection"""
        
        if self.collection_stats['quality_scores']:
            avg_quality = sum(self.collection_stats['quality_scores']) / len(self.collection_stats['quality_scores'])
        else:
            avg_quality = 0.0
        
        return {
            'total_records_collected': self.collection_stats['total_records'],
            'sources_attempted': len(self.collection_stats['sources_attempted']),
            'sources_successful': len(self.collection_stats['sources_successful']),
            'success_rate': len(self.collection_stats['sources_successful']) / max(1, len(self.collection_stats['sources_attempted'])),
            'average_quality_score': avg_quality,
            'successful_sources': self.collection_stats['sources_successful'],
            'timestamp': datetime.utcnow().isoformat()
        }


# Example usage
async def main():
    """Test the enhanced data sources"""
    
    logging.basicConfig(level=logging.INFO)
    
    async with aiohttp.ClientSession() as session:
        manager = EnhancedDataSourceManager(session)
        
        # Collect from all sources
        all_data = await manager.collect_all_enhanced_sources()
        
        # Show summary
        summary = manager.get_collection_summary()
        
        print("=== Enhanced Data Source Collection Summary ===")
        print(f"Total records: {summary['total_records_collected']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Average quality: {summary['average_quality_score']:.2f}")
        print(f"Successful sources: {', '.join(summary['successful_sources'])}")
        
        # Show sample data from each source
        for source_name, data_list in all_data.items():
            if data_list:
                print(f"\n{source_name.upper()} - Sample record:")
                sample = data_list[0]
                print(f"  Title: {sample.get('title', 'N/A')}")
                print(f"  Source: {sample.get('source', 'N/A')}")
                if 'quality_assessment' in sample:
                    qa = sample['quality_assessment']
                    print(f"  Quality: {qa['score']:.1f} ({qa['level']})")


if __name__ == "__main__":
    asyncio.run(main())