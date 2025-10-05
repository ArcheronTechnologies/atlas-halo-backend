#!/usr/bin/env python3
"""
RSS Feed Collector for Swedish Police Data
Collects incident data from regional police RSS feeds to supplement the main API
Part of Atlas AI data ingestion pipeline
"""

import asyncio
import aiohttp
import feedparser
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

@dataclass
class RSSIncident:
    """RSS incident from Swedish police feed"""
    id: str
    title: str
    description: str
    link: str
    published: datetime
    region: str
    raw_entry: Dict[str, Any]

class SwedishPoliceRSSCollector:
    """
    Collector for Swedish Police Regional RSS Feeds
    Supplements the main Polisen API with RSS feed data
    """
    
    BASE_URL = "https://polisen.se"
    
    # Regional RSS feed paths - updated from polisen.se/aktuellt/rss/lokala-rss-floden/
    REGIONAL_RSS_FEEDS = {
        'stockholm': '/aktuellt/rss/stockholms-lan/handelser-rss---stockholms-lan/',
        'vastra_gotaland': '/aktuellt/rss/vastra-gotaland/handelser-rss---vastra-gotaland/',
        'skane': '/aktuellt/rss/skane/handelser-rss---skane/',
        'blekinge': '/aktuellt/rss/blekinge/handelser-rss---blekinge/',
        'dalarna': '/aktuellt/rss/dalarna/handelser-rss---dalarna/',
        'gotland': '/aktuellt/rss/gotland/handelser-rss---gotland/',
        'gavleborg': '/aktuellt/rss/gavleborg/handelser-rss---gavleborg/',
        'halland': '/aktuellt/rss/halland/handelser-rss---halland/',
        'jamtland': '/aktuellt/rss/jamtland/handelser-rss---jamtland/',
        'jonkoping': '/aktuellt/rss/jonkoping-lan/handelser-rss---jonkoping-lan/',  # Fixed: add -lan
        'kalmar': '/aktuellt/rss/kalmar-lan/handelser-rss---kalmar-lan/',  # Fixed: add -lan
        'kronoberg': '/aktuellt/rss/kronoberg/handelser-rss---kronoberg/',
        'norrbotten': '/aktuellt/rss/norrbotten/handelser-rss---norrbotten/',
        'orebro': '/aktuellt/rss/orebro-lan/handelser-rss---orebro-lan/',  # Fixed: add -lan
        'ostergotland': '/aktuellt/rss/ostergotland/handelser-rss---ostergotland/',
        'sodermanland': '/aktuellt/rss/sodermanland/handelser-rss---sodermanland/',
        'uppsala': '/aktuellt/rss/uppsala-lan/handelser-rss---uppsala-lan/',  # Fixed: add -lan
        'varmland': '/aktuellt/rss/varmland/handelser-rss---varmland/',
        'vasterbotten': '/aktuellt/rss/vasterbotten/handelser-rss---vasterbotten/',
        'vasternorrland': '/aktuellt/rss/vasternorrland/handelser-rss---vasternorrland/',
        'vastmanland': '/aktuellt/rss/vastmanland/handelser-rss---vastmanland/'
    }
    
    # Fallback URLs in case primary URLs fail
    FALLBACK_FEEDS = {
        'jonkoping': ['/aktuellt/rss/jonkoping/handelser-rss---jonkoping/', 
                     '/aktuellt/rss/jonkoping-lan/handelser-rss---jonkoping-lan/'],
        'kalmar': ['/aktuellt/rss/kalmar/handelser-rss---kalmar/',
                  '/aktuellt/rss/kalmar-lan/handelser-rss---kalmar-lan/'],
        'orebro': ['/aktuellt/rss/orebro/handelser-rss---orebro/',
                  '/aktuellt/rss/orebro-lan/handelser-rss---orebro-lan/'],
        'uppsala': ['/aktuellt/rss/uppsala/handelser-rss---uppsala/',
                   '/aktuellt/rss/uppsala-lan/handelser-rss---uppsala-lan/']
    }
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session or aiohttp.ClientSession()
        self.rate_limit_delay = 2.0  # Be respectful to RSS feeds
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_rss_feed(self, region: str, feed_path: str) -> List[RSSIncident]:
        """Fetch and parse RSS feed for a specific region with fallback support"""
        
        urls_to_try = [feed_path]
        
        # Add fallback URLs if available
        if region in self.FALLBACK_FEEDS:
            urls_to_try.extend(self.FALLBACK_FEEDS[region])
        
        for url_path in urls_to_try:
            url = urljoin(self.BASE_URL, url_path)
            
            try:
                logger.info(f"Fetching RSS feed for {region}: {url}")
                
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse RSS feed using feedparser
                        feed = feedparser.parse(content)
                        
                        if not feed.entries:
                            logger.warning(f"No entries found in RSS feed for {region}")
                            continue  # Try next URL
                        
                        incidents = []
                        for entry in feed.entries:
                            try:
                                incident = self._parse_rss_entry(entry, region)
                                if incident:
                                    incidents.append(incident)
                            except Exception as e:
                                logger.warning(f"Failed to parse RSS entry: {e}")
                                continue
                        
                        logger.info(f"Retrieved {len(incidents)} incidents from {region} RSS feed")
                        return incidents
                        
                    elif response.status == 404:
                        logger.warning(f"RSS feed not found for {region}: {response.status} - trying fallback")
                        continue  # Try next URL
                    else:
                        logger.error(f"RSS request failed for {region}: {response.status}")
                        continue  # Try next URL
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching RSS feed for {region}: {url}")
                continue  # Try next URL
            except Exception as e:
                logger.error(f"Error fetching RSS feed for {region}: {e}")
                continue  # Try next URL
            
            finally:
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
        
        logger.error(f"All RSS feed URLs failed for {region}")
        return []
    
    def _parse_rss_entry(self, entry: Any, region: str) -> Optional[RSSIncident]:
        """Parse individual RSS entry into RSSIncident"""
        
        try:
            # Extract published date
            published = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])
            else:
                # Fallback to current time
                published = datetime.utcnow()
            
            # Generate unique ID from link or title+date
            incident_id = entry.get('id') or entry.get('link', '')
            if not incident_id:
                incident_id = f"{region}_{hash(entry.title + str(published))}"
            
            return RSSIncident(
                id=incident_id,
                title=entry.get('title', ''),
                description=entry.get('description') or entry.get('summary', ''),
                link=entry.get('link', ''),
                published=published,
                region=region,
                raw_entry={
                    'title': entry.get('title'),
                    'description': entry.get('description'),
                    'summary': entry.get('summary'),
                    'link': entry.get('link'),
                    'published': entry.get('published'),
                    'updated': entry.get('updated'),
                    'id': entry.get('id')
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    async def collect_all_regions(self, 
                                 hours_back: int = 24,
                                 regions: List[str] = None) -> Dict[str, List[RSSIncident]]:
        """Collect incidents from all or specified regions"""
        
        regions_to_fetch = regions or list(self.REGIONAL_RSS_FEEDS.keys())
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        results = {}
        
        for region in regions_to_fetch:
            if region not in self.REGIONAL_RSS_FEEDS:
                logger.warning(f"Unknown region: {region}")
                continue
                
            feed_path = self.REGIONAL_RSS_FEEDS[region]
            incidents = await self.fetch_rss_feed(region, feed_path)
            
            # Filter by time window
            recent_incidents = [
                incident for incident in incidents 
                if incident.published >= cutoff_time
            ]
            
            results[region] = recent_incidents
            
        return results
    
    async def collect_major_cities(self, hours_back: int = 24) -> Dict[str, List[RSSIncident]]:
        """Collect incidents from major Swedish cities/regions"""
        
        major_regions = ['stockholm', 'vastra_gotaland', 'skane']  # Stockholm, Gothenburg, Malmö regions
        return await self.collect_all_regions(hours_back=hours_back, regions=major_regions)
    
    def extract_location_info(self, incident: RSSIncident) -> Dict[str, Any]:
        """Extract location information from RSS incident"""
        
        location_info = {
            'region': incident.region,
            'city': None,
            'address': None,
            'coordinates': None
        }
        
        # Try to extract city/location from title or description
        text_to_search = f"{incident.title} {incident.description}".lower()
        
        # Common Swedish cities/areas
        city_patterns = {
            'stockholm': r'stockholm',
            'göteborg': r'g[öo]teborg',
            'malmö': r'malm[öo]',
            'uppsala': r'uppsala',
            'linköping': r'link[öo]ping',
            'västerås': r'v[äa]ster[åa]s',
            'helsingborg': r'helsingborg',
            'norrköping': r'norrk[öo]ping',
            'lund': r'lund',
            'umeå': r'ume[åa]'
        }
        
        for city, pattern in city_patterns.items():
            if re.search(pattern, text_to_search):
                location_info['city'] = city
                break
        
        # Try to extract street/address information
        # Look for Swedish address patterns (e.g., "Kungsgatan 15", "Storgatan")
        address_match = re.search(r'([A-ZÅÄÖ][a-zåäö]+(?:gatan|vägen|torget|plats))\s*\d*', 
                                incident.title + ' ' + incident.description)
        if address_match:
            location_info['address'] = address_match.group(1)
        
        return location_info
    
    def classify_incident_type(self, incident: RSSIncident) -> str:
        """Classify incident type from RSS data"""
        
        text = f"{incident.title} {incident.description}".lower()
        
        # Crime type classification based on Swedish terms
        if any(term in text for term in ['rån', 'rånade']):
            return 'robbery'
        elif any(term in text for term in ['stöld', 'stulit', 'inbrott']):
            return 'theft'
        elif any(term in text for term in ['misshandel', 'våld', 'slagsmål']):
            return 'assault'
        elif any(term in text for term in ['skadegörelse', 'vandalism']):
            return 'vandalism'
        elif any(term in text for term in ['narkotika', 'droger']):
            return 'drug_offense'
        elif any(term in text for term in ['trafikolycka', 'olycka']):
            return 'traffic_accident'
        elif any(term in text for term in ['brand']):
            return 'fire'
        elif any(term in text for term in ['bedrägeri']):
            return 'fraud'
        else:
            return 'other'
    
    def process_incident(self, incident: RSSIncident) -> Dict[str, Any]:
        """Process RSS incident into standardized format"""
        
        return {
            'id': f"rss_{incident.id}",
            'title': incident.title,
            'description': incident.description,
            'crime_type': self.classify_incident_type(incident),
            'severity_score': self._calculate_severity(incident),
            'datetime': incident.published.isoformat(),
            'location': self.extract_location_info(incident),
            'source': 'polisen_rss',
            'source_region': incident.region,
            'url': incident.link,
            'processed_at': datetime.utcnow().isoformat(),
            'raw_data': incident.raw_entry
        }
    
    def _calculate_severity(self, incident: RSSIncident) -> int:
        """Calculate severity score for RSS incident"""
        
        crime_type = self.classify_incident_type(incident)
        base_severity = {
            'robbery': 5,
            'assault': 4,
            'drug_offense': 3,
            'theft': 3,
            'vandalism': 2,
            'fraud': 2,
            'traffic_accident': 2,
            'fire': 4,
            'other': 2
        }.get(crime_type, 2)
        
        # Adjust based on keywords in title/description
        text = f"{incident.title} {incident.description}".lower()
        
        if any(word in text for word in ['vapen', 'skott', 'kniv']):
            base_severity += 2
        
        if any(word in text for word in ['våldtäkt', 'mord']):
            base_severity = 5
            
        return min(base_severity, 5)


# Additional municipal data source connectors
class MunicipalDataConnector:
    """Connector for municipal open data portals"""
    
    MUNICIPAL_ENDPOINTS = {
        'malmo': {
            'police_incidents': 'https://polisen.se/api/events?locationname=Malmö',
            'name': 'Malmö Police Incidents'
        }
        # Add other municipalities as they become available
    }
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session or aiohttp.ClientSession()
    
    async def fetch_malmo_incidents(self) -> List[Dict[str, Any]]:
        """Fetch incidents specific to Malmö"""
        
        url = self.MUNICIPAL_ENDPOINTS['malmo']['police_incidents']
        
        try:
            logger.info(f"Fetching Malmö incidents from: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter for Malmö-specific incidents
                    malmo_incidents = []
                    for incident in data:
                        location_name = incident.get('location', {}).get('name', '').lower()
                        if 'malmö' in location_name or 'malmo' in location_name:
                            malmo_incidents.append(incident)
                    
                    logger.info(f"Found {len(malmo_incidents)} Malmö-specific incidents")
                    return malmo_incidents
                    
                else:
                    logger.error(f"Failed to fetch Malmö incidents: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching Malmö incidents: {e}")
            return []


# Example usage
async def main():
    """Test the RSS feed collector"""
    
    logging.basicConfig(level=logging.INFO)
    
    async with SwedishPoliceRSSCollector() as collector:
        
        # Test major cities collection
        print("Collecting incidents from major cities...")
        major_city_incidents = await collector.collect_major_cities(hours_back=48)
        
        total_incidents = sum(len(incidents) for incidents in major_city_incidents.values())
        print(f"Total incidents from major cities: {total_incidents}")
        
        for region, incidents in major_city_incidents.items():
            print(f"\n{region.upper()}: {len(incidents)} incidents")
            
            for incident in incidents[:3]:  # Show first 3 per region
                processed = collector.process_incident(incident)
                print(f"  - {processed['title']}")
                print(f"    Type: {processed['crime_type']}, Severity: {processed['severity_score']}")
                print(f"    Location: {processed['location']['city'] or 'Unknown city'}")
    
    # Test municipal connector
    async with MunicipalDataConnector() as municipal:
        malmo_incidents = await municipal.fetch_malmo_incidents()
        print(f"\nMalmö municipal incidents: {len(malmo_incidents)}")


if __name__ == "__main__":
    asyncio.run(main())