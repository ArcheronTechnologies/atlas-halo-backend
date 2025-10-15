"""
Web Intelligence Gathering System

This module provides comprehensive web intelligence capabilities for monitoring
the global electronics supply chain, including news analysis, supplier monitoring,
and market intelligence aggregation from internet sources.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import json
import hashlib
import re
from urllib.parse import urljoin, urlparse
# Optional imports for web intelligence features
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    BeautifulSoup = None
    HAS_BS4 = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    feedparser = None
    HAS_FEEDPARSER = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    TextBlob = None
    HAS_TEXTBLOB = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    yf = None
    HAS_YFINANCE = False

from ..cache.redis_cache import cache
from ..db.mongo import get_mongo_db

logger = logging.getLogger(__name__)


@dataclass
class WebIntelligenceSource:
    """Configuration for a web intelligence source"""
    name: str
    url: str
    source_type: str  # news, supplier, market, social
    crawl_frequency: int  # minutes
    selectors: Dict[str, str]  # CSS selectors for content extraction
    keywords: List[str]
    region: str = "global"
    language: str = "en"
    priority: int = 1
    active: bool = True


@dataclass
class IntelligenceItem:
    """A piece of web intelligence"""
    id: str
    source: str
    title: str
    content: str
    url: str
    published_at: datetime
    discovered_at: datetime
    item_type: str  # news, announcement, alert, analysis
    confidence: float  # 0.0 to 1.0
    sentiment: float  # -1.0 to 1.0 
    relevance_score: float  # 0.0 to 1.0
    keywords: List[str]
    entities: Dict[str, List[str]]  # companies, components, locations
    geopolitical_impact: Dict[str, Any]
    market_impact: Dict[str, Any]
    region: str
    language: str


class NewsAggregator:
    """Aggregates news from multiple sources"""
    
    def __init__(self):
        self.sources = [
            # Electronics industry news
            WebIntelligenceSource(
                name="Electronics Weekly",
                url="https://www.electronicsweekly.com/feed/",
                source_type="news",
                crawl_frequency=30,
                selectors={"title": "title", "content": "description"},
                keywords=["semiconductor", "chip", "component", "supply chain", "shortage"]
            ),
            WebIntelligenceSource(
                name="EE Times",
                url="https://www.eetimes.com/feed/",
                source_type="news", 
                crawl_frequency=30,
                selectors={"title": "title", "content": "description"},
                keywords=["electronics", "semiconductor", "manufacturing", "supply chain"]
            ),
            # Financial news
            WebIntelligenceSource(
                name="Reuters Technology",
                url="https://feeds.reuters.com/reuters/technologyNews",
                source_type="news",
                crawl_frequency=15,
                selectors={"title": "title", "content": "description"},
                keywords=["semiconductor", "chip shortage", "supply chain", "manufacturing"]
            ),
            WebIntelligenceSource(
                name="Bloomberg Technology",
                url="https://feeds.bloomberg.com/technology/news.rss",
                source_type="news",
                crawl_frequency=15,
                selectors={"title": "title", "content": "description"},
                keywords=["semiconductor", "chip", "supply chain", "electronics"]
            ),
            # Trade publications
            WebIntelligenceSource(
                name="Supply Chain Dive",
                url="https://www.supplychaindive.com/feeds/news/",
                source_type="news",
                crawl_frequency=60,
                selectors={"title": "title", "content": "description"},
                keywords=["electronics", "semiconductor", "components", "shortage", "disruption"]
            )
        ]
        
        self.processed_items: Set[str] = set()
        self.session = None
    
    async def start_monitoring(self):
        """Start continuous news monitoring"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'SCIP Intelligence Bot 1.0'}
        )
        
        # Start monitoring tasks for each source
        tasks = []
        for source in self.sources:
            if source.active:
                task = asyncio.create_task(self._monitor_source(source))
                tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _monitor_source(self, source: WebIntelligenceSource):
        """Monitor a single news source continuously"""
        while True:
            try:
                await self._crawl_source(source)
                await asyncio.sleep(source.crawl_frequency * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring source {source.name}: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _crawl_source(self, source: WebIntelligenceSource):
        """Crawl a single news source"""
        try:
            async with self.session.get(source.url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {source.name}: HTTP {response.status}")
                    return
                
                content = await response.text()
                
                if source.url.endswith('.xml') or 'rss' in source.url or 'feed' in source.url:
                    # RSS/Atom feed
                    items = await self._parse_rss_feed(content, source)
                else:
                    # Regular webpage
                    items = await self._parse_webpage(content, source)
                
                # Process and store items
                for item in items:
                    if item.id not in self.processed_items:
                        await self._process_intelligence_item(item)
                        self.processed_items.add(item.id)
                        
                        # Keep processed items set manageable
                        if len(self.processed_items) > 10000:
                            # Remove oldest 2000 items
                            self.processed_items = set(list(self.processed_items)[2000:])
                
                logger.info(f"Processed {len(items)} items from {source.name}")
                
        except Exception as e:
            logger.error(f"Error crawling source {source.name}: {e}")
    
    async def _parse_rss_feed(self, content: str, source: WebIntelligenceSource) -> List[IntelligenceItem]:
        """Parse RSS/Atom feed content"""
        items = []
        
        if not HAS_FEEDPARSER:
            logger.warning("feedparser not available, skipping RSS parsing")
            return items
        
        try:
            feed = feedparser.parse(content)
            
            for entry in feed.entries[:10]:  # Process last 10 items
                # Check if content is relevant
                text = f"{entry.get('title', '')} {entry.get('description', '')}"
                if not self._is_relevant_content(text, source.keywords):
                    continue
                
                # Extract published date
                published_at = datetime.now(timezone.utc)
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except:
                        pass
                
                # Create intelligence item
                item_id = hashlib.md5(f"{source.name}:{entry.get('link', '')}".encode()).hexdigest()
                
                item = IntelligenceItem(
                    id=item_id,
                    source=source.name,
                    title=entry.get('title', ''),
                    content=entry.get('description', ''),
                    url=entry.get('link', ''),
                    published_at=published_at,
                    discovered_at=datetime.now(timezone.utc),
                    item_type='news',
                    confidence=0.8,
                    sentiment=0.0,  # Will be analyzed later
                    relevance_score=0.0,  # Will be calculated later
                    keywords=[],
                    entities={},
                    geopolitical_impact={},
                    market_impact={},
                    region=source.region,
                    language=source.language
                )
                
                items.append(item)
                
        except Exception as e:
            logger.error(f"Error parsing RSS feed from {source.name}: {e}")
        
        return items
    
    async def _parse_webpage(self, content: str, source: WebIntelligenceSource) -> List[IntelligenceItem]:
        """Parse regular webpage content"""
        items = []
        
        if not HAS_BS4:
            logger.warning("BeautifulSoup not available, skipping webpage parsing")
            return items
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract articles/news items based on common patterns
            article_selectors = [
                'article',
                '.article',
                '.news-item',
                '.post',
                '.entry'
            ]
            
            articles = []
            for selector in article_selectors:
                articles.extend(soup.select(selector)[:5])  # Max 5 per selector
            
            for article in articles[:10]:  # Process max 10 articles
                title_elem = article.find(['h1', 'h2', 'h3', '.title', '.headline'])
                content_elem = article.find(['p', '.content', '.summary', '.excerpt'])
                link_elem = article.find('a')
                
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                content = content_elem.get_text(strip=True) if content_elem else ""
                url = urljoin(source.url, link_elem.get('href', '')) if link_elem else source.url
                
                # Check relevance
                text = f"{title} {content}"
                if not self._is_relevant_content(text, source.keywords):
                    continue
                
                # Create intelligence item
                item_id = hashlib.md5(f"{source.name}:{url}".encode()).hexdigest()
                
                item = IntelligenceItem(
                    id=item_id,
                    source=source.name,
                    title=title,
                    content=content,
                    url=url,
                    published_at=datetime.now(timezone.utc),  # Unknown, use current time
                    discovered_at=datetime.now(timezone.utc),
                    item_type='news',
                    confidence=0.7,
                    sentiment=0.0,
                    relevance_score=0.0,
                    keywords=[],
                    entities={},
                    geopolitical_impact={},
                    market_impact={},
                    region=source.region,
                    language=source.language
                )
                
                items.append(item)
                
        except Exception as e:
            logger.error(f"Error parsing webpage from {source.name}: {e}")
        
        return items
    
    def _is_relevant_content(self, text: str, keywords: List[str]) -> bool:
        """Check if content is relevant based on keywords"""
        text_lower = text.lower()
        
        # Must match at least one keyword
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return True
        
        return False
    
    async def _process_intelligence_item(self, item: IntelligenceItem):
        """Process and enrich an intelligence item"""
        try:
            # Perform sentiment analysis
            if item.content and HAS_TEXTBLOB:
                blob = TextBlob(item.content)
                item.sentiment = blob.sentiment.polarity
            else:
                item.sentiment = 0.0  # Neutral if TextBlob not available
            
            # Extract entities and keywords
            item.keywords = self._extract_keywords(f"{item.title} {item.content}")
            item.entities = self._extract_entities(f"{item.title} {item.content}")
            
            # Calculate relevance score
            item.relevance_score = self._calculate_relevance_score(item)
            
            # Analyze geopolitical impact
            item.geopolitical_impact = await self._analyze_geopolitical_impact(item)
            
            # Analyze market impact
            item.market_impact = await self._analyze_market_impact(item)
            
            # Store in database
            await self._store_intelligence_item(item)
            
            # Cache high-relevance items
            if item.relevance_score > 0.7:
                await cache.set('high_relevance_intel', item.id, asdict(item), ttl=86400)
            
            logger.debug(f"Processed intelligence item: {item.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing intelligence item {item.id}: {e}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Electronics industry keywords
        industry_terms = [
            'semiconductor', 'chip', 'microprocessor', 'memory', 'flash',
            'component', 'resistor', 'capacitor', 'inductor', 'transistor',
            'supply chain', 'shortage', 'allocation', 'lead time',
            'manufacturing', 'fab', 'foundry', 'assembly',
            'automotive', 'industrial', 'consumer', 'medical'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in industry_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        # Extract quoted terms and proper nouns
        quoted_terms = re.findall(r'"([^"]*)"', text)
        found_keywords.extend(quoted_terms)
        
        return list(set(found_keywords))[:10]  # Limit to 10 keywords
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'companies': [],
            'components': [],
            'locations': [],
            'people': []
        }
        
        # Common company patterns
        company_patterns = [
            r'\b(Intel|AMD|NVIDIA|Qualcomm|Broadcom|Texas Instruments|Analog Devices)\b',
            r'\b(TSMC|Samsung|SK Hynix|Micron|Western Digital)\b',
            r'\b(Apple|Google|Microsoft|Amazon|Tesla)\b',
            r'\b[A-Z][a-zA-Z]+ (Corp|Corporation|Inc|Incorporated|Ltd|Limited|AG|GmbH)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['companies'].extend(matches)
        
        # Component patterns
        component_patterns = [
            r'\b\d+[nμm]m (process|node)\b',
            r'\bDDR\d+ (SDRAM|memory)\b',
            r'\b(ARM|x86|RISC-V) (processor|core)\b',
            r'\b\d+GB (flash|memory|storage)\b'
        ]
        
        for pattern in component_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['components'].extend(matches)
        
        # Location patterns
        location_patterns = [
            r'\b(China|Taiwan|South Korea|Japan|United States|Europe)\b',
            r'\b(Beijing|Shanghai|Shenzhen|Taipei|Seoul|Tokyo|Silicon Valley)\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['locations'].extend(matches)
        
        # Clean up and deduplicate
        for key in entities:
            entities[key] = list(set([e.strip() for e in entities[key] if e.strip()]))[:5]
        
        return entities
    
    def _calculate_relevance_score(self, item: IntelligenceItem) -> float:
        """Calculate relevance score for an intelligence item"""
        score = 0.0
        
        # Keyword relevance
        high_value_keywords = [
            'shortage', 'supply chain', 'disruption', 'allocation',
            'lead time', 'price increase', 'manufacturing halt'
        ]
        
        for keyword in item.keywords:
            if keyword.lower() in high_value_keywords:
                score += 0.2
            else:
                score += 0.1
        
        # Entity relevance
        if item.entities.get('companies'):
            score += 0.3
        if item.entities.get('components'):
            score += 0.3
        if item.entities.get('locations'):
            score += 0.1
        
        # Sentiment impact
        if abs(item.sentiment) > 0.5:  # Strong sentiment
            score += 0.2
        
        # Time relevance (newer is more relevant)
        hours_since_published = (item.discovered_at - item.published_at).total_seconds() / 3600
        if hours_since_published < 24:
            score += 0.2
        elif hours_since_published < 168:  # 1 week
            score += 0.1
        
        return min(score, 1.0)
    
    async def _analyze_geopolitical_impact(self, item: IntelligenceItem) -> Dict[str, Any]:
        """Analyze geopolitical impact of the intelligence item"""
        impact = {
            'risk_level': 'low',  # low, medium, high, critical
            'affected_regions': [],
            'policy_implications': [],
            'trade_impact': False,
            'sanction_risk': False
        }
        
        text = f"{item.title} {item.content}".lower()
        
        # High-risk keywords
        if any(keyword in text for keyword in ['sanction', 'trade war', 'export ban', 'embargo']):
            impact['risk_level'] = 'critical'
            impact['trade_impact'] = True
            impact['sanction_risk'] = True
        
        # Medium-risk keywords
        elif any(keyword in text for keyword in ['tariff', 'trade dispute', 'restriction', 'regulation']):
            impact['risk_level'] = 'medium'
            impact['trade_impact'] = True
        
        # Policy implications
        if any(keyword in text for keyword in ['policy', 'regulation', 'compliance', 'standard']):
            impact['policy_implications'].append('regulatory_change')
        
        # Affected regions
        regions = ['china', 'taiwan', 'south korea', 'japan', 'europe', 'united states']
        impact['affected_regions'] = [region for region in regions if region in text]
        
        return impact
    
    async def _analyze_market_impact(self, item: IntelligenceItem) -> Dict[str, Any]:
        """Analyze market impact of the intelligence item"""
        impact = {
            'price_impact': 'neutral',  # positive, negative, neutral
            'supply_impact': 'stable',  # shortage, surplus, stable
            'demand_impact': 'stable',  # high, low, stable
            'affected_categories': [],
            'timeline': 'unknown'  # immediate, short_term, medium_term, long_term
        }
        
        text = f"{item.title} {item.content}".lower()
        
        # Price impact analysis
        if any(keyword in text for keyword in ['price increase', 'cost rise', 'expensive']):
            impact['price_impact'] = 'negative'
        elif any(keyword in text for keyword in ['price drop', 'cost reduction', 'cheaper']):
            impact['price_impact'] = 'positive'
        
        # Supply impact analysis
        if any(keyword in text for keyword in ['shortage', 'supply constraint', 'allocation']):
            impact['supply_impact'] = 'shortage'
        elif any(keyword in text for keyword in ['oversupply', 'excess inventory', 'surplus']):
            impact['supply_impact'] = 'surplus'
        
        # Demand impact analysis
        if any(keyword in text for keyword in ['strong demand', 'high demand', 'increased orders']):
            impact['demand_impact'] = 'high'
        elif any(keyword in text for keyword in ['weak demand', 'low demand', 'reduced orders']):
            impact['demand_impact'] = 'low'
        
        # Timeline analysis
        if any(keyword in text for keyword in ['immediate', 'now', 'today']):
            impact['timeline'] = 'immediate'
        elif any(keyword in text for keyword in ['next month', 'coming weeks', 'soon']):
            impact['timeline'] = 'short_term'
        elif any(keyword in text for keyword in ['next quarter', 'months ahead']):
            impact['timeline'] = 'medium_term'
        elif any(keyword in text for keyword in ['next year', 'long term', 'years']):
            impact['timeline'] = 'long_term'
        
        # Affected categories
        categories = ['semiconductor', 'memory', 'automotive', 'industrial', 'consumer']
        impact['affected_categories'] = [cat for cat in categories if cat in text]
        
        return impact
    
    async def _store_intelligence_item(self, item: IntelligenceItem):
        """Store intelligence item in database"""
        try:
            mongo = get_mongo_db()
            if mongo:
                collection = mongo.get_collection("web_intelligence")
                
                # Convert to dict for storage
                item_dict = asdict(item)
                item_dict['published_at'] = item.published_at.isoformat()
                item_dict['discovered_at'] = item.discovered_at.isoformat()
                
                # Upsert to avoid duplicates
                await collection.update_one(
                    {'id': item.id},
                    {'$set': item_dict},
                    upsert=True
                )
        except Exception as e:
            logger.error(f"Error storing intelligence item {item.id}: {e}")
    
    async def stop_monitoring(self):
        """Stop news monitoring"""
        if self.session:
            await self.session.close()


class SupplierMonitor:
    """Monitors supplier websites for announcements and changes"""
    
    def __init__(self):
        self.supplier_sources = [
            # Major semiconductor suppliers
            WebIntelligenceSource(
                name="Intel Newsroom",
                url="https://newsroom.intel.com/news-releases/",
                source_type="supplier",
                crawl_frequency=120,  # 2 hours
                selectors={"title": ".news-title", "content": ".news-summary"},
                keywords=["product", "announcement", "availability", "EOL", "shortage"]
            ),
            WebIntelligenceSource(
                name="TI News",
                url="https://news.ti.com/",
                source_type="supplier",
                crawl_frequency=240,  # 4 hours
                selectors={"title": "h1", "content": ".content"},
                keywords=["product", "announcement", "discontinuation", "new"]
            )
        ]
    
    async def start_monitoring(self):
        """Start supplier monitoring"""
        # Similar implementation to NewsAggregator
        # Focus on product announcements, EOL notices, supply updates
        pass


class MarketDataCollector:
    """Collects market data from financial and trading sources"""
    
    def __init__(self):
        self.tracked_symbols = [
            'INTC', 'AMD', 'NVDA', 'QCOM', 'BRCM', 'TXN', 'ADI',  # US semiconductors
            'TSM', '2330.TW',  # Taiwan semiconductors 
            '005930.KS', '000660.KS',  # Korean semiconductors
            'ASML', 'ASMI.AS'  # European equipment
        ]
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """Collect real-time market data"""
        market_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stocks': {},
            'indices': {},
            'commodities': {}
        }
        
        try:
            # Collect stock data
            for symbol in self.tracked_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    history = ticker.history(period='5d')
                    
                    if not history.empty:
                        latest = history.iloc[-1]
                        market_data['stocks'][symbol] = {
                            'price': float(latest['Close']),
                            'change': float(latest['Close'] - history.iloc[-2]['Close']),
                            'change_percent': float((latest['Close'] - history.iloc[-2]['Close']) / history.iloc[-2]['Close'] * 100),
                            'volume': int(latest['Volume']),
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0)
                        }
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {e}")
            
            # Store in cache for quick access
            await cache.set('market_data', 'current', market_data, ttl=300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
        
        return market_data
    
    async def analyze_market_sentiment(self) -> Dict[str, Any]:
        """Analyze overall market sentiment for electronics sector"""
        try:
            market_data = await cache.get('market_data', 'current')
            if not market_data:
                market_data = await self.collect_market_data()
            
            # Calculate sector performance
            stocks = market_data.get('stocks', {})
            if not stocks:
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            changes = [stock['change_percent'] for stock in stocks.values()]
            avg_change = sum(changes) / len(changes)
            
            # Determine sentiment
            if avg_change > 2:
                sentiment = 'very_positive'
            elif avg_change > 0.5:
                sentiment = 'positive'
            elif avg_change > -0.5:
                sentiment = 'neutral'
            elif avg_change > -2:
                sentiment = 'negative'
            else:
                sentiment = 'very_negative'
            
            # Calculate confidence based on consistency
            positive_count = sum(1 for change in changes if change > 0)
            confidence = abs(positive_count / len(changes) - 0.5) * 2
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'avg_change_percent': avg_change,
                'stocks_positive': positive_count,
                'stocks_total': len(changes),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}


class WebIntelligenceOrchestrator:
    """Orchestrates all web intelligence gathering activities"""
    
    def __init__(self):
        self.news_aggregator = NewsAggregator()
        self.supplier_monitor = SupplierMonitor()
        self.market_collector = MarketDataCollector()
        self.running = False
        self.tasks = []
    
    async def start_intelligence_gathering(self):
        """Start all intelligence gathering processes"""
        if self.running:
            logger.warning("Intelligence gathering already running")
            return
        
        self.running = True
        logger.info("Starting web intelligence gathering...")
        
        # Start news monitoring
        news_task = asyncio.create_task(self.news_aggregator.start_monitoring())
        self.tasks.append(news_task)
        
        # Start market data collection (every 5 minutes)
        market_task = asyncio.create_task(self._market_data_loop())
        self.tasks.append(market_task)
        
        # Start supplier monitoring
        supplier_task = asyncio.create_task(self.supplier_monitor.start_monitoring())
        self.tasks.append(supplier_task)
        
        logger.info("Web intelligence gathering started")
    
    async def _market_data_loop(self):
        """Continuous market data collection"""
        while self.running:
            try:
                await self.market_collector.collect_market_data()
                await asyncio.sleep(300)  # 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data collection: {e}")
                await asyncio.sleep(300)
    
    async def get_intelligence_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get intelligence summary for the specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        summary = {
            'period_hours': hours_back,
            'news_items': 0,
            'high_relevance_items': 0,
            'geopolitical_alerts': 0,
            'market_sentiment': {},
            'top_keywords': [],
            'affected_regions': [],
            'supply_chain_alerts': [],
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            mongo = get_mongo_db()
            if mongo:
                collection = mongo.get_collection("web_intelligence")
                
                # Query recent items
                cursor = collection.find({
                    'discovered_at': {'$gte': cutoff_time.isoformat()}
                })
                
                items = await cursor.to_list(length=1000)  # Max 1000 items
                summary['news_items'] = len(items)
                
                # Analyze items
                keyword_counts = {}
                regions = set()
                supply_alerts = []
                
                for item in items:
                    # Count relevance
                    if item.get('relevance_score', 0) > 0.7:
                        summary['high_relevance_items'] += 1
                    
                    # Count geopolitical alerts
                    geo_impact = item.get('geopolitical_impact', {})
                    if geo_impact.get('risk_level') in ['high', 'critical']:
                        summary['geopolitical_alerts'] += 1
                    
                    # Collect keywords
                    for keyword in item.get('keywords', []):
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                    
                    # Collect regions
                    regions.update(geo_impact.get('affected_regions', []))
                    
                    # Supply chain alerts
                    market_impact = item.get('market_impact', {})
                    if market_impact.get('supply_impact') == 'shortage':
                        supply_alerts.append({
                            'title': item.get('title'),
                            'categories': market_impact.get('affected_categories', []),
                            'timeline': market_impact.get('timeline')
                        })
                
                # Top keywords
                summary['top_keywords'] = sorted(keyword_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]
                summary['affected_regions'] = list(regions)
                summary['supply_chain_alerts'] = supply_alerts
            
            # Get market sentiment
            summary['market_sentiment'] = await self.market_collector.analyze_market_sentiment()
            
        except Exception as e:
            logger.error(f"Error generating intelligence summary: {e}")
        
        return summary
    
    async def search_intelligence(self, 
                                query: str, 
                                item_type: Optional[str] = None,
                                days_back: int = 7) -> List[Dict[str, Any]]:
        """Search intelligence items"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        try:
            mongo = get_mongo_db()
            if not mongo:
                return []
            
            collection = mongo.get_collection("web_intelligence")
            
            # Build search filter
            search_filter = {
                'discovered_at': {'$gte': cutoff_time.isoformat()},
                '$or': [
                    {'title': {'$regex': query, '$options': 'i'}},
                    {'content': {'$regex': query, '$options': 'i'}},
                    {'keywords': {'$in': [query.lower()]}}
                ]
            }
            
            if item_type:
                search_filter['item_type'] = item_type
            
            cursor = collection.find(search_filter).sort('relevance_score', -1).limit(50)
            items = await cursor.to_list(length=50)
            
            return items
            
        except Exception as e:
            logger.error(f"Error searching intelligence: {e}")
            return []
    
    async def stop_intelligence_gathering(self):
        """Stop all intelligence gathering processes"""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop individual components
        await self.news_aggregator.stop_monitoring()
        
        logger.info("Web intelligence gathering stopped")


# Global orchestrator instance
web_intelligence = WebIntelligenceOrchestrator()


# API helper functions
async def start_web_intelligence():
    """Start web intelligence gathering"""
    await web_intelligence.start_intelligence_gathering()


async def stop_web_intelligence():
    """Stop web intelligence gathering"""
    await web_intelligence.stop_intelligence_gathering()


async def get_intelligence_dashboard() -> Dict[str, Any]:
    """Get intelligence dashboard data"""
    return await web_intelligence.get_intelligence_summary(hours_back=24)


async def search_web_intelligence(query: str, days_back: int = 7) -> List[Dict[str, Any]]:
    """Search web intelligence"""
    return await web_intelligence.search_intelligence(query, days_back=days_back)