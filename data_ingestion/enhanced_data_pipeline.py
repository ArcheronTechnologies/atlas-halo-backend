#!/usr/bin/env python3
"""
Enhanced Data Ingestion Pipeline for Atlas AI
Combines multiple Swedish crime data sources for comprehensive training dataset
Integrates: Polisen API, RSS feeds, and municipal data sources
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import os
from pathlib import Path

# Import our existing collectors
import sys
from pathlib import Path

# Add the path to atlas_mvp backend
atlas_backend_path = Path(__file__).parent.parent.parent / "atlas_mvp" / "backend" / "src"
if atlas_backend_path.exists():
    sys.path.insert(0, str(atlas_backend_path))

try:
    from polisen_api import PolisenAPIClient, CrimeDataProcessor
except ImportError:
    # If polisen_api is not available, create minimal stubs
    class PolisenAPIClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get_recent_events(self, hours_back=24): return []
    
    class CrimeDataProcessor:
        def process_incident(self, incident): return incident

try:
    from .rss_feed_collector import SwedishPoliceRSSCollector, MunicipalDataConnector
    from .data_quality_validator import EnhancedDataQualityValidator
    from .adaptive_scheduler import AdaptiveCollectionScheduler
    from .enhanced_data_sources import EnhancedDataSourceManager
except ImportError:
    # Fallback for direct execution
    from rss_feed_collector import SwedishPoliceRSSCollector, MunicipalDataConnector
    from data_quality_validator import EnhancedDataQualityValidator
    from adaptive_scheduler import AdaptiveCollectionScheduler
    from enhanced_data_sources import EnhancedDataSourceManager

logger = logging.getLogger(__name__)

class EnhancedDataPipeline:
    """
    Enhanced data pipeline that combines multiple Swedish crime data sources
    for comprehensive AI training dataset
    """
    
    def __init__(self, output_dir: str = "data_lake/enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize enhanced components
        self.validator = EnhancedDataQualityValidator()
        self.scheduler = AdaptiveCollectionScheduler()
        self.enhanced_sources = None  # Will be initialized with session
        
        # Initialize collectors
        self.polisen_client = None
        self.rss_collector = None
        self.municipal_connector = None
        self.crime_processor = CrimeDataProcessor()
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.polisen_client = PolisenAPIClient()
        self.rss_collector = SwedishPoliceRSSCollector()
        self.municipal_connector = MunicipalDataConnector()
        self.enhanced_sources = EnhancedDataSourceManager()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.polisen_client:
            await self.polisen_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.rss_collector:
            await self.rss_collector.__aexit__(exc_type, exc_val, exc_tb)
        if self.municipal_connector:
            await self.municipal_connector.session.close()
    
    async def collect_all_sources(self, hours_back: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect data from all available sources
        
        Returns:
            Dictionary with data from each source
        """
        
        results = {
            'polisen_api': [],
            'rss_feeds': [],
            'municipal_data': [],
            'metadata': {
                'collection_time': datetime.utcnow().isoformat(),
                'hours_back': hours_back,
                'sources_attempted': ['polisen_api', 'rss_feeds', 'municipal_data']
            }
        }
        
        logger.info(f"Starting enhanced data collection for last {hours_back} hours")
        
        # 1. Collect from main Polisen API
        try:
            logger.info("Collecting from Polisen API...")
            polisen_incidents = await self.polisen_client.get_recent_events(hours_back=hours_back)
            
            # Process incidents
            for incident in polisen_incidents:
                processed = self.crime_processor.process_incident(incident)
                results['polisen_api'].append(processed)
                
            logger.info(f"Collected {len(results['polisen_api'])} incidents from Polisen API")
            
        except Exception as e:
            logger.error(f"Failed to collect from Polisen API: {e}")
        
        # 2. Collect from RSS feeds (all regions)
        try:
            logger.info("Collecting from regional RSS feeds...")
            rss_data = await self.rss_collector.collect_all_regions(hours_back=hours_back)
            
            # Process RSS incidents
            for region, incidents in rss_data.items():
                for incident in incidents:
                    processed = self.rss_collector.process_incident(incident)
                    results['rss_feeds'].append(processed)
            
            total_rss = len(results['rss_feeds'])
            logger.info(f"Collected {total_rss} incidents from RSS feeds across {len(rss_data)} regions")
            
        except Exception as e:
            logger.error(f"Failed to collect from RSS feeds: {e}")
        
        # 3. Collect from municipal sources
        try:
            logger.info("Collecting from municipal sources...")
            
            # Malmö incidents (legacy connector)
            malmo_incidents = await self.municipal_connector.fetch_malmo_incidents()
            
            # Convert to standard format
            for incident in malmo_incidents:
                # These are already in Polisen format, just add source info
                processed = incident.copy()
                processed['source'] = 'municipal_malmo'
                processed['processed_at'] = datetime.utcnow().isoformat()
                results['municipal_data'].append(processed)
            
            logger.info(f"Collected {len(results['municipal_data'])} municipal incidents")
            
        except Exception as e:
            logger.error(f"Failed to collect from municipal sources: {e}")
        
        # 4. Collect from enhanced data sources (BRÅ, Stockholm, etc.)
        try:
            if self.enhanced_sources:
                logger.info("Collecting from enhanced data sources...")
                enhanced_data = await self.enhanced_sources.collect_all_enhanced_sources(hours_back)
                
                # Merge enhanced sources into results
                for source_name, data_list in enhanced_data.items():
                    if source_name not in results:
                        results[source_name] = []
                    results[source_name].extend(data_list)
                
                enhanced_count = sum(len(data_list) for data_list in enhanced_data.values())
                logger.info(f"Collected {enhanced_count} records from enhanced sources")
                
        except Exception as e:
            logger.error(f"Failed to collect from enhanced sources: {e}")
        
        # 5. Apply quality validation to all collected data
        try:
            logger.info("Applying quality validation...")
            quality_assessments = []
            
            for source_name, incidents in results.items():
                if source_name == 'metadata':
                    continue
                    
                for incident in incidents:
                    assessment = self.validator.validate_incident(incident)
                    incident['quality_score'] = assessment.score
                    incident['quality_level'] = assessment.level.name
                    incident['quality_issues'] = assessment.issues
                    incident['duplicate_risk'] = assessment.duplicate_risk
                    quality_assessments.append(assessment)
            
            # Record results for adaptive scheduling
            collection_time = 5.0  # Placeholder timing
            self.scheduler.record_collection_result(
                incidents_collected=sum(len(incidents) for key, incidents in results.items() if key != 'metadata'),
                quality_assessments=quality_assessments,
                collection_time_seconds=collection_time,
                errors=[]
            )
            
            logger.info(f"Quality validation complete: {len(quality_assessments)} assessments")
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
        
        # Summary
        total_incidents = sum(len(incidents) for key, incidents in results.items() if key != 'metadata')
        
        results['metadata']['total_incidents'] = total_incidents
        results['metadata']['source_counts'] = {
            'polisen_api': len(results['polisen_api']),
            'rss_feeds': len(results['rss_feeds']),
            'municipal_data': len(results['municipal_data'])
        }
        
        logger.info(f"Enhanced collection complete: {total_incidents} total incidents")
        return results
    
    def deduplicate_incidents(self, all_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Deduplicate incidents across sources using various strategies
        """
        
        all_incidents = []
        seen_ids = set()
        location_time_cache = {}
        
        # Collect all incidents
        for source, incidents in all_data.items():
            if source == 'metadata':
                continue
                
            for incident in incidents:
                # Add source info if not present
                if 'source' not in incident:
                    incident['source'] = source
                all_incidents.append(incident)
        
        logger.info(f"Deduplicating {len(all_incidents)} incidents...")
        
        deduplicated = []
        
        for incident in all_incidents:
            # Strategy 1: Exact ID match
            incident_id = incident.get('id', '')
            if incident_id in seen_ids:
                logger.debug(f"Skipping duplicate ID: {incident_id}")
                continue
            
            # Strategy 2: Location + time similarity
            location_name = incident.get('location', {}).get('name', '') or incident.get('title', '')
            incident_time = incident.get('datetime', '')
            
            # Create location-time key for fuzzy matching
            location_key = self._normalize_location(location_name)
            time_key = self._normalize_time(incident_time)
            location_time_key = f"{location_key}_{time_key}"
            
            if location_time_key in location_time_cache:
                # Check if this is likely the same incident
                existing = location_time_cache[location_time_key]
                similarity = self._calculate_similarity(incident, existing)
                
                if similarity > 0.7:  # High similarity threshold
                    logger.debug(f"Skipping likely duplicate: {location_time_key}")
                    continue
            
            # This appears to be a unique incident
            seen_ids.add(incident_id)
            location_time_cache[location_time_key] = incident
            deduplicated.append(incident)
        
        logger.info(f"Deduplication complete: {len(deduplicated)} unique incidents (removed {len(all_incidents) - len(deduplicated)} duplicates)")
        return deduplicated
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location string for comparison"""
        if not location:
            return ""
        
        # Remove common prefixes and normalize
        location = location.lower()
        location = location.replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
        
        # Remove common words that don't help with matching
        common_words = ['kommun', 'stad', 'län', 'polisstation']
        for word in common_words:
            location = location.replace(word, '')
        
        return location.strip()
    
    def _normalize_time(self, time_str: str) -> str:
        """Normalize time for comparison (to nearest hour)"""
        if not time_str:
            return ""
        
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            # Round to nearest hour for comparison
            return dt.strftime('%Y-%m-%d_%H')
        except:
            return ""
    
    def _calculate_similarity(self, incident1: Dict[str, Any], incident2: Dict[str, Any]) -> float:
        """Calculate similarity between two incidents"""
        
        # Title/description similarity
        title1 = incident1.get('title', '') or incident1.get('original_name', '')
        title2 = incident2.get('title', '') or incident2.get('original_name', '')
        
        # Simple word overlap similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
    
    async def save_collected_data(self, data: Dict[str, Any], filename_prefix: str = None) -> str:
        """Save collected data to disk"""
        
        if not filename_prefix:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename_prefix = f"enhanced_collection_{timestamp}"
        
        # Save raw data
        raw_file = self.output_dir / f"{filename_prefix}_raw.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        # Deduplicate and save processed data
        deduplicated = self.deduplicate_incidents(data)
        
        processed_data = {
            'incidents': deduplicated,
            'metadata': data['metadata'],
            'deduplication_stats': {
                'original_count': data['metadata']['total_incidents'],
                'deduplicated_count': len(deduplicated),
                'duplicates_removed': data['metadata']['total_incidents'] - len(deduplicated)
            }
        }
        
        processed_file = self.output_dir / f"{filename_prefix}_processed.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Data saved to {raw_file} and {processed_file}")
        return str(processed_file)
    
    def generate_training_stats(self, processed_file: str) -> Dict[str, Any]:
        """Generate statistics about collected data for training purposes"""
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        incidents = data['incidents']
        
        stats = {
            'total_incidents': len(incidents),
            'date_range': self._calculate_date_range(incidents),
            'source_distribution': self._calculate_source_distribution(incidents),
            'crime_type_distribution': self._calculate_crime_type_distribution(incidents),
            'location_distribution': self._calculate_location_distribution(incidents),
            'severity_distribution': self._calculate_severity_distribution(incidents),
            'temporal_distribution': self._calculate_temporal_distribution(incidents)
        }
        
        return stats
    
    def _calculate_date_range(self, incidents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate the date range of incidents"""
        
        dates = []
        for incident in incidents:
            date_str = incident.get('datetime')
            if date_str:
                try:
                    # Handle both timezone-aware and naive datetimes
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    # Convert to UTC if not already timezone-aware
                    if dt.tzinfo is None:
                        from datetime import timezone
                        dt = dt.replace(tzinfo=timezone.utc)
                    dates.append(dt)
                except:
                    continue
        
        if dates:
            return {
                'earliest': min(dates).isoformat(),
                'latest': max(dates).isoformat(),
                'span_hours': (max(dates) - min(dates)).total_seconds() / 3600
            }
        
        return {'earliest': None, 'latest': None, 'span_hours': 0}
    
    def _calculate_source_distribution(self, incidents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution by data source"""
        
        distribution = {}
        for incident in incidents:
            source = incident.get('source', 'unknown')
            distribution[source] = distribution.get(source, 0) + 1
        
        return distribution
    
    def _calculate_crime_type_distribution(self, incidents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution by crime type"""
        
        distribution = {}
        for incident in incidents:
            crime_type = incident.get('crime_type', 'unknown')
            distribution[crime_type] = distribution.get(crime_type, 0) + 1
        
        return distribution
    
    def _calculate_location_distribution(self, incidents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution by location/region"""
        
        distribution = {}
        for incident in incidents:
            # Try different location fields
            location = (incident.get('location', {}).get('city') or 
                       incident.get('location', {}).get('region') or
                       incident.get('source_region') or
                       'unknown')
            distribution[location] = distribution.get(location, 0) + 1
        
        return distribution
    
    def _calculate_severity_distribution(self, incidents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution by severity score"""
        
        distribution = {}
        for incident in incidents:
            severity = incident.get('severity_score', 'unknown')
            distribution[str(severity)] = distribution.get(str(severity), 0) + 1
        
        return distribution
    
    def _calculate_temporal_distribution(self, incidents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution by hour of day"""
        
        distribution = {}
        for incident in incidents:
            date_str = incident.get('datetime')
            if date_str:
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    hour = dt.hour
                    distribution[str(hour)] = distribution.get(str(hour), 0) + 1
                except:
                    continue
        
        return distribution


# Main execution function
async def run_enhanced_collection(hours_back: int = 48, save_results: bool = True) -> Dict[str, Any]:
    """
    Run the enhanced data collection pipeline
    
    Args:
        hours_back: How many hours back to collect data
        save_results: Whether to save results to disk
        
    Returns:
        Collection results and statistics
    """
    
    logging.basicConfig(level=logging.INFO)
    
    async with EnhancedDataPipeline() as pipeline:
        
        # Collect data from all sources
        all_data = await pipeline.collect_all_sources(hours_back=hours_back)
        
        # Save data if requested
        processed_file = None
        if save_results:
            processed_file = await pipeline.save_collected_data(all_data)
        
        # Generate statistics
        if processed_file:
            stats = pipeline.generate_training_stats(processed_file)
            
            # Save stats
            stats_file = processed_file.replace('_processed.json', '_stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Training statistics saved to {stats_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("ENHANCED DATA COLLECTION SUMMARY")
            print("="*60)
            print(f"Total incidents collected: {stats['total_incidents']}")
            print(f"Date range: {stats['date_range']['span_hours']:.1f} hours")
            print("\nSource distribution:")
            for source, count in stats['source_distribution'].items():
                print(f"  {source}: {count} incidents")
            print("\nCrime type distribution:")
            for crime_type, count in stats['crime_type_distribution'].items():
                print(f"  {crime_type}: {count} incidents")
            print("\nLocation distribution (top 5):")
            sorted_locations = sorted(stats['location_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
            for location, count in sorted_locations:
                print(f"  {location}: {count} incidents")
            print("="*60)
            
            return {
                'data_file': processed_file,
                'stats': stats,
                'metadata': all_data['metadata']
            }
        
        return {'data': all_data}


# Example usage
if __name__ == "__main__":
    
    # Run enhanced collection
    results = asyncio.run(run_enhanced_collection(hours_back=72, save_results=True))
    
    if 'stats' in results:
        print(f"\nData collection complete!")
        print(f"Results saved to: {results['data_file']}")
        print(f"Total incidents: {results['stats']['total_incidents']}")
    else:
        print("Data collection completed but not saved")