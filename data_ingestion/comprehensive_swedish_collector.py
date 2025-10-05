"""
Comprehensive Swedish Crime Data Collector
Enhanced data collection for Atlas AI training with larger datasets
"""

import asyncio
import aiohttp
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time

from .historical_polisen_data import HistoricalPolisenDataIngestion, HistoricalCrimeRecord
from .bra_pxweb_client import build_bra_reported_offences_query, ingest_bra_table, make_month_codes
from ..database.postgis_database import get_database

logger = logging.getLogger(__name__)


@dataclass
class CollectionTarget:
    """Data collection targets for comprehensive training data"""
    total_records: int = 50000
    min_quality_score: float = 0.6
    years_back: int = 3
    cities_coverage: List[str] = None
    temporal_distribution_target: Dict[str, float] = None
    
    def __post_init__(self):
        if self.cities_coverage is None:
            self.cities_coverage = ["Stockholm", "Gothenburg", "Malmö", "Uppsala", "Västerås", "Örebro"]
        
        if self.temporal_distribution_target is None:
            self.temporal_distribution_target = {
                "recent_6_months": 0.4,
                "6_12_months": 0.3,
                "1_2_years": 0.2,
                "2_3_years": 0.1
            }


class ComprehensiveSwedishCollector:
    """
    Comprehensive Swedish crime data collector for Atlas AI training
    
    Features:
    - Multi-source Swedish crime data collection
    - Intelligent data quality assessment
    - Temporal and spatial coverage optimization
    - Real-time progress tracking
    - Dataset validation and statistics
    """
    
    def __init__(self, target: CollectionTarget = None):
        self.target = target or CollectionTarget()
        self.database = None
        self.collected_records = []
        self.collection_stats = {
            "start_time": None,
            "end_time": None,
            "sources_attempted": [],
            "sources_successful": [],
            "records_by_source": {},
            "records_by_city": {},
            "records_by_year": {},
            "quality_distribution": {},
            "errors": []
        }
        
        # Enhanced Swedish locations with precise coordinates
        self.swedish_locations = {
            "Stockholm": {
                "county": "Stockholms län",
                "center": (59.3293, 18.0686),
                "radius_km": 30,
                "municipalities": ["Stockholm", "Solna", "Sundbyberg", "Nacka", "Huddinge", "Södermalm"],
                "target_weight": 0.25  # 25% of total records
            },
            "Gothenburg": {
                "county": "Västra Götalands län",
                "center": (57.7089, 11.9746),
                "radius_km": 25,
                "municipalities": ["Göteborg", "Mölndal", "Partille", "Härryda", "Lerum"],
                "target_weight": 0.20  # 20% of total records
            },
            "Malmö": {
                "county": "Skåne län",
                "center": (55.6050, 13.0038),
                "radius_km": 20,
                "municipalities": ["Malmö", "Lund", "Burlöv", "Staffanstorp", "Svedala"],
                "target_weight": 0.15
            },
            "Uppsala": {
                "county": "Uppsala län",
                "center": (59.8586, 17.6389),
                "radius_km": 15,
                "municipalities": ["Uppsala", "Knivsta", "Håbo", "Enköping"],
                "target_weight": 0.10
            },
            "Västerås": {
                "county": "Västmanlands län",
                "center": (59.6162, 16.5528),
                "radius_km": 15,
                "municipalities": ["Västerås", "Hallstahammar", "Surahammar"],
                "target_weight": 0.10
            },
            "Örebro": {
                "county": "Örebro län",
                "center": (59.2741, 15.2066),
                "radius_km": 15,
                "municipalities": ["Örebro", "Kumla", "Hallsberg"],
                "target_weight": 0.10
            },
            "Norrköping": {
                "county": "Östergötlands län",
                "center": (58.5942, 16.1826),
                "radius_km": 15,
                "municipalities": ["Norrköping", "Söderköping", "Finspång"],
                "target_weight": 0.10
            }
        }
    
    async def initialize(self):
        """Initialize the comprehensive collector"""
        self.database = await get_database()
        self.collection_stats["start_time"] = datetime.now()
        logger.info("🚀 Comprehensive Swedish Crime Data Collector initialized")
    
    async def collect_comprehensive_training_data(self) -> Dict[str, Any]:
        """
        Collect comprehensive training data meeting all targets
        """
        logger.info("📊 Starting comprehensive Swedish training data collection...")
        logger.info(f"Target: {self.target.total_records:,} records across {len(self.target.cities_coverage)} cities")
        
        # Phase 1: Historical Polisen.se data collection
        await self._collect_enhanced_polisen_data()
        
        # Phase 2: Augment with additional Swedish sources
        await self._collect_additional_swedish_sources()
        
        # Phase 3: Validate and optimize dataset
        await self._validate_and_optimize_dataset()
        
        # Phase 4: Store and generate statistics
        final_stats = await self._finalize_collection()
        
        return final_stats
    
    async def _collect_enhanced_polisen_data(self):
        """Enhanced Polisen.se data collection with intelligent targeting"""
        logger.info("🇸🇪 Collecting enhanced Polisen.se data...")
        
        ingestion_system = HistoricalPolisenDataIngestion()
        await ingestion_system.initialize()
        
        self.collection_stats["sources_attempted"].append("polisen_se")
        
        try:
            # Calculate target per city based on weights
            city_targets = {}
            for city, config in self.swedish_locations.items():
                if city in self.target.cities_coverage:
                    city_targets[city] = int(self.target.total_records * config["target_weight"])
            
            # Collect data for each city with temporal distribution
            for city_name, target_count in city_targets.items():
                logger.info(f"🏙️ Collecting {target_count:,} records from {city_name}...")
                
                city_records = await self._collect_city_data_temporal(
                    ingestion_system, city_name, target_count
                )
                
                self.collected_records.extend(city_records)
                self.collection_stats["records_by_city"][city_name] = len(city_records)
                
                logger.info(f"✅ {city_name}: {len(city_records)} records collected")
            
            self.collection_stats["sources_successful"].append("polisen_se")
            self.collection_stats["records_by_source"]["polisen_se"] = len(self.collected_records)
            
        except Exception as e:
            error_msg = f"Polisen.se collection failed: {e}"
            logger.error(error_msg)
            self.collection_stats["errors"].append(error_msg)
        
        finally:
            await ingestion_system.cleanup()
    
    async def _collect_city_data_temporal(
        self, 
        ingestion_system: HistoricalPolisenDataIngestion,
        city_name: str,
        target_count: int
    ) -> List[HistoricalCrimeRecord]:
        """Collect city data with proper temporal distribution"""
        
        city_config = self.swedish_locations[city_name]
        all_city_records = []
        
        # Calculate temporal targets
        temporal_targets = {}
        for period, weight in self.target.temporal_distribution_target.items():
            temporal_targets[period] = int(target_count * weight)
        
        # Collect for each temporal period
        for period, period_target in temporal_targets.items():
            start_date, end_date = self._get_period_dates(period)
            
            logger.debug(f"Collecting {period_target} records from {city_name} for {period}")
            
            # Use multiple collection strategies for better coverage
            period_records = []
            
            # Strategy 1: County-level collection
            county_records = await self._collect_by_location(
                ingestion_system, "län", city_config["county"], 
                start_date, end_date, period_target // 2
            )
            period_records.extend(county_records)
            
            # Strategy 2: Municipality-level collection  
            remaining_target = period_target - len(period_records)
            if remaining_target > 0:
                for municipality in city_config["municipalities"]:
                    muni_target = remaining_target // len(city_config["municipalities"])
                    muni_records = await self._collect_by_location(
                        ingestion_system, "kommun", municipality,
                        start_date, end_date, muni_target
                    )
                    period_records.extend(muni_records)
                    
                    if len(period_records) >= period_target:
                        break
            
            # Add period metadata
            for record in period_records:
                if hasattr(record, '__dict__'):
                    record.__dict__['_collection_period'] = period
                    record.__dict__['_collection_city'] = city_name
            
            all_city_records.extend(period_records[:period_target])
        
        return all_city_records
    
    async def _collect_by_location(
        self,
        ingestion_system: HistoricalPolisenDataIngestion,
        location_type: str,
        location_name: str,
        start_date: datetime,
        end_date: datetime,
        target_count: int
    ) -> List[HistoricalCrimeRecord]:
        """Collect records by specific location and time range"""
        
        try:
            # Use the enhanced collection method
            raw_records = await ingestion_system.fetch_events_by_location_and_time(
                location=location_name,
                start_date=start_date,
                end_date=end_date,
                max_records=target_count * 2  # Collect extra to filter for quality
            )
            
            # Process and validate records
            processed_records = []
            for raw_record in raw_records:
                processed_record = await ingestion_system._process_and_validate_record(raw_record)
                if (processed_record and 
                    processed_record.data_quality_score >= self.target.min_quality_score):
                    processed_records.append(processed_record)
                    
                    # Track year distribution
                    year = processed_record.datetime_occurred.year
                    if year not in self.collection_stats["records_by_year"]:
                        self.collection_stats["records_by_year"][year] = 0
                    self.collection_stats["records_by_year"][year] += 1
                
                if len(processed_records) >= target_count:
                    break
            
            return processed_records
            
        except Exception as e:
            logger.error(f"Error collecting from {location_name}: {e}")
            return []
    
    def _get_period_dates(self, period: str) -> Tuple[datetime, datetime]:
        """Get start and end dates for temporal period"""
        now = datetime.now(timezone.utc)
        
        if period == "recent_6_months":
            start_date = now - timedelta(days=180)
            end_date = now
        elif period == "6_12_months":
            start_date = now - timedelta(days=365)
            end_date = now - timedelta(days=180)
        elif period == "1_2_years":
            start_date = now - timedelta(days=730)
            end_date = now - timedelta(days=365)
        elif period == "2_3_years":
            start_date = now - timedelta(days=1095)
            end_date = now - timedelta(days=730)
        else:
            # Default to last year
            start_date = now - timedelta(days=365)
            end_date = now
        
        return start_date, end_date
    
    async def _collect_additional_swedish_sources(self):
        """Collect from additional Swedish crime data sources"""
        logger.info("📈 Collecting from additional Swedish sources...")
        
        # Brå PxWeb: monthly reported offences (national total) for target years
        try:
            years_back = max(2, min(5, self.target.years_back))
            now_ts = datetime.now(timezone.utc)
            periods = make_month_codes(
                now_ts.year - years_back,
                now_ts.year,
                end_month=now_ts.month,
            )
            query = build_bra_reported_offences_query(region_codes=["00"], offence_codes=None, periods=periods)
            rows = await ingest_bra_table("Nationella_brottsstatistik/b1201", query)
            if rows:
                # Store via database if available
                if not self.database:
                    self.database = await get_database()
                inserted = await self.database.store_bra_statistics_rows(rows)
                logger.info(f"✅ Inserted {inserted} Brå monthly statistic rows")
            else:
                logger.warning("No Brå rows returned during comprehensive collection")
        except Exception as e:
            logger.warning(f"Brå ingestion skipped or failed: {e}")
    
    async def _validate_and_optimize_dataset(self):
        """Validate dataset quality and optimize for training"""
        logger.info("🔍 Validating and optimizing dataset...")
        
        # Remove duplicates
        initial_count = len(self.collected_records)
        unique_records = {}
        
        for record in self.collected_records:
            # Create unique key based on location, time, and crime type
            key = f"{record.polisen_id}_{record.latitude}_{record.longitude}_{record.datetime_occurred.isoformat()}"
            
            if key not in unique_records or record.data_quality_score > unique_records[key].data_quality_score:
                unique_records[key] = record
        
        self.collected_records = list(unique_records.values())
        duplicates_removed = initial_count - len(self.collected_records)
        
        if duplicates_removed > 0:
            logger.info(f"🧹 Removed {duplicates_removed} duplicate records")
        
        # Calculate quality distribution
        quality_bins = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
        
        for record in self.collected_records:
            score = record.data_quality_score
            if score < 0.3:
                quality_bins["0.0-0.3"] += 1
            elif score < 0.5:
                quality_bins["0.3-0.5"] += 1
            elif score < 0.7:
                quality_bins["0.5-0.7"] += 1
            elif score < 0.9:
                quality_bins["0.7-0.9"] += 1
            else:
                quality_bins["0.9-1.0"] += 1
        
        self.collection_stats["quality_distribution"] = quality_bins
        
        # Filter for minimum quality
        high_quality_records = [
            record for record in self.collected_records 
            if record.data_quality_score >= self.target.min_quality_score
        ]
        
        logger.info(f"📊 Dataset validation results:")
        logger.info(f"  Total records: {len(self.collected_records):,}")
        logger.info(f"  High quality (>={self.target.min_quality_score}): {len(high_quality_records):,}")
        logger.info(f"  Quality distribution: {quality_bins}")
        
        self.collected_records = high_quality_records
    
    async def _finalize_collection(self) -> Dict[str, Any]:
        """Finalize collection and generate comprehensive statistics"""
        logger.info("🏁 Finalizing collection...")
        
        self.collection_stats["end_time"] = datetime.now()
        duration = self.collection_stats["end_time"] - self.collection_stats["start_time"]
        
        # Store records in database
        if self.collected_records:
            await self._bulk_store_records()
        
        # Generate comprehensive statistics
        final_stats = {
            "collection_summary": {
                "total_records_collected": len(self.collected_records),
                "target_records": self.target.total_records,
                "target_achievement": (len(self.collected_records) / self.target.total_records * 100) if self.target.total_records > 0 else 0,
                "collection_duration_minutes": duration.total_seconds() / 60,
                "records_per_minute": len(self.collected_records) / (duration.total_seconds() / 60) if duration.total_seconds() > 0 else 0
            },
            "data_quality": {
                "average_quality_score": sum(r.data_quality_score for r in self.collected_records) / len(self.collected_records) if self.collected_records else 0,
                "quality_distribution": self.collection_stats["quality_distribution"],
                "high_quality_percentage": len([r for r in self.collected_records if r.data_quality_score >= 0.8]) / len(self.collected_records) * 100 if self.collected_records else 0
            },
            "coverage_analysis": {
                "cities_covered": list(self.collection_stats["records_by_city"].keys()),
                "records_by_city": self.collection_stats["records_by_city"],
                "temporal_coverage_years": list(self.collection_stats["records_by_year"].keys()),
                "records_by_year": self.collection_stats["records_by_year"]
            },
            "sources": {
                "sources_attempted": self.collection_stats["sources_attempted"],
                "sources_successful": self.collection_stats["sources_successful"],
                "records_by_source": self.collection_stats["records_by_source"]
            },
            "training_readiness": {
                "ready_for_training": len(self.collected_records) >= 10000,
                "recommended_next_steps": self._get_training_recommendations()
            },
            "errors": self.collection_stats["errors"]
        }
        
        logger.info("✅ Collection completed successfully!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"  Records collected: {final_stats['collection_summary']['total_records_collected']:,}")
        logger.info(f"  Target achievement: {final_stats['collection_summary']['target_achievement']:.1f}%")
        logger.info(f"  Average quality: {final_stats['data_quality']['average_quality_score']:.3f}")
        logger.info(f"  Cities covered: {len(final_stats['coverage_analysis']['cities_covered'])}")
        logger.info(f"  Training ready: {'✅ Yes' if final_stats['training_readiness']['ready_for_training'] else '❌ No'}")
        
        return final_stats
    
    async def _bulk_store_records(self):
        """Bulk store collected records in database"""
        try:
            # Use the existing bulk storage method
            ingestion_system = HistoricalPolisenDataIngestion(self.database)
            await ingestion_system._bulk_store_historical_records(self.collected_records)
            logger.info(f"💾 Stored {len(self.collected_records)} records in database")
        except Exception as e:
            logger.error(f"Error storing records: {e}")
            self.collection_stats["errors"].append(f"Storage error: {e}")
    
    def _get_training_recommendations(self) -> List[str]:
        """Get recommendations for training based on collection results"""
        recommendations = []
        
        total_records = len(self.collected_records)
        
        if total_records < 10000:
            recommendations.append("Collect more data - minimum 10,000 records recommended for training")
        elif total_records < 25000:
            recommendations.append("Consider collecting additional data for better model performance")
        
        avg_quality = sum(r.data_quality_score for r in self.collected_records) / len(self.collected_records) if self.collected_records else 0
        if avg_quality < 0.7:
            recommendations.append("Improve data quality - focus on records with coordinates and detailed descriptions")
        
        cities_count = len(set(getattr(r, '_collection_city', 'Unknown') for r in self.collected_records))
        if cities_count < 3:
            recommendations.append("Add more geographic diversity - collect from additional Swedish cities")
        
        years_count = len(set(r.datetime_occurred.year for r in self.collected_records))
        if years_count < 2:
            recommendations.append("Add temporal diversity - collect data from multiple years")
        
        if not recommendations:
            recommendations.append("Dataset is ready for training - proceed with model development")
        
        return recommendations


# CLI interface for comprehensive collection
async def run_comprehensive_collection():
    """Run comprehensive Swedish crime data collection"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🇸🇪 Atlas AI - Comprehensive Swedish Crime Data Collection")
    print("========================================================")
    
    # Configure collection target
    target = CollectionTarget(
        total_records=50000,
        min_quality_score=0.6,
        years_back=3,
        cities_coverage=["Stockholm", "Göteborg", "Malmö", "Uppsala", "Västerås", "Örebro", "Norrköping"]
    )
    
    # Initialize collector
    collector = ComprehensiveSwedishCollector(target)
    await collector.initialize()
    
    # Run collection
    results = await collector.collect_comprehensive_training_data()
    
    # Display results
    print("\n📊 Collection Results:")
    print(f"  Total Records: {results['collection_summary']['total_records_collected']:,}")
    print(f"  Target Achievement: {results['collection_summary']['target_achievement']:.1f}%")
    print(f"  Duration: {results['collection_summary']['collection_duration_minutes']:.1f} minutes")
    print(f"  Average Quality: {results['data_quality']['average_quality_score']:.3f}")
    
    print(f"\n🏙️ Coverage by City:")
    for city, count in results['coverage_analysis']['records_by_city'].items():
        print(f"  {city}: {count:,} records")
    
    print(f"\n📅 Coverage by Year:")
    for year, count in results['coverage_analysis']['records_by_year'].items():
        print(f"  {year}: {count:,} records")
    
    print(f"\n🎯 Training Readiness:")
    print(f"  Ready: {'✅ Yes' if results['training_readiness']['ready_for_training'] else '❌ No'}")
    print(f"  Recommendations:")
    for rec in results['training_readiness']['recommended_next_steps']:
        print(f"    - {rec}")
    
    if results['errors']:
        print(f"\n⚠️ Errors ({len(results['errors'])}):")
        for error in results['errors'][-3:]:  # Show last 3 errors
            print(f"    - {error}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_collection())
