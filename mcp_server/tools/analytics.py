"""
Advanced Data Analytics Tools

Provides advanced analytics functionality including trend analysis, platform comparison, keyword co-occurrence, sentiment analysis, etc.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from difflib import SequenceMatcher

from ..services.data_service import DataService
from ..utils.validators import (
    validate_platforms,
    validate_limit,
    validate_keyword,
    validate_top_n,
    validate_date_range
)
from ..utils.errors import MCPError, InvalidParameterError, DataNotFoundError


def calculate_news_weight(news_data: Dict, rank_threshold: int = 5) -> float:
    """
    Calculate news weight (for sorting)

    Based on the weight algorithm implementation in main.py, comprehensively considers:
    - Rank weight (60%): News ranking in the list
    - Frequency weight (30%): Number of times news appears
    - Hotness weight (10%): Proportion of high rankings

    Args:
        news_data: News data dictionary, containing ranks and count fields
        rank_threshold: High rank threshold, default 5

    Returns:
        Weight score (float between 0-100)
    """
    ranks = news_data.get("ranks", [])
    if not ranks:
        return 0.0

    count = news_data.get("count", len(ranks))

    # Weight configuration (consistent with config.yaml)
    RANK_WEIGHT = 0.6
    FREQUENCY_WEIGHT = 0.3
    HOTNESS_WEIGHT = 0.1

    # 1. Rank weight: Σ(11 - min(rank, 10)) / occurrence count
    rank_scores = []
    for rank in ranks:
        score = 11 - min(rank, 10)
        rank_scores.append(score)

    rank_weight = sum(rank_scores) / len(ranks) if ranks else 0

    # 2. Frequency weight: min(occurrence count, 10) × 10
    frequency_weight = min(count, 10) * 10

    # 3. Hotness bonus: high rank count / total occurrence count × 100
    high_rank_count = sum(1 for rank in ranks if rank <= rank_threshold)
    hotness_ratio = high_rank_count / len(ranks) if ranks else 0
    hotness_weight = hotness_ratio * 100

    # Combined weight
    total_weight = (
        rank_weight * RANK_WEIGHT
        + frequency_weight * FREQUENCY_WEIGHT
        + hotness_weight * HOTNESS_WEIGHT
    )

    return total_weight


class AnalyticsTools:
    """Advanced Data Analytics Tools Class"""

    def __init__(self, project_root: str = None):
        """
        Initialize analytics tools

        Args:
            project_root: Project root directory
        """
        self.data_service = DataService(project_root)

    def analyze_data_insights_unified(
        self,
        insight_type: str = "platform_compare",
        topic: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None,
        min_frequency: int = 3,
        top_n: int = 20
    ) -> Dict:
        """
        Unified data insights analysis tool - integrates multiple data analysis modes

        Args:
            insight_type: Insight type, optional values:
                - "platform_compare": Platform comparison analysis (compare different platforms' attention to topics)
                - "platform_activity": Platform activity statistics (statistics on each platform's publishing frequency and active time)
                - "keyword_cooccur": Keyword co-occurrence analysis (analyze patterns of keywords appearing together)
            topic: Topic keyword (optional, applicable for platform_compare mode)
            date_range: Date range, format: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
            min_frequency: Minimum co-occurrence frequency (for keyword_cooccur mode), default 3
            top_n: Return TOP N results (for keyword_cooccur mode), default 20

        Returns:
            Data insights analysis result dictionary

        Examples:
            - analyze_data_insights_unified(insight_type="platform_compare", topic="AI")
            - analyze_data_insights_unified(insight_type="platform_activity", date_range={...})
            - analyze_data_insights_unified(insight_type="keyword_cooccur", min_frequency=5)
        """
        try:
            # Parameter validation
            if insight_type not in ["platform_compare", "platform_activity", "keyword_cooccur"]:
                raise InvalidParameterError(
                    f"Invalid insight type: {insight_type}",
                    suggestion="Supported types: platform_compare, platform_activity, keyword_cooccur"
                )

            # Call corresponding method based on insight type
            if insight_type == "platform_compare":
                return self.compare_platforms(
                    topic=topic,
                    date_range=date_range
                )
            elif insight_type == "platform_activity":
                return self.get_platform_activity_stats(
                    date_range=date_range
                )
            else:  # keyword_cooccur
                return self.analyze_keyword_cooccurrence(
                    min_frequency=min_frequency,
                    top_n=top_n
                )

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def analyze_topic_trend_unified(
        self,
        topic: str,
        analysis_type: str = "trend",
        date_range: Optional[Dict[str, str]] = None,
        granularity: str = "day",
        threshold: float = 3.0,
        time_window: int = 24,
        lookahead_hours: int = 6,
        confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Unified topic trend analysis tool - integrates multiple trend analysis modes

        Args:
            topic: Topic keyword (required)
            analysis_type: Analysis type, optional values:
                - "trend": Hotness trend analysis (track topic's hotness changes)
                - "lifecycle": Lifecycle analysis (complete cycle from appearance to disappearance)
                - "viral": Abnormal hotness detection (identify suddenly trending topics)
                - "predict": Topic prediction (predict future possible hot topics)
            date_range: Date range (for trend and lifecycle modes), optional
                       - **Format**: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
                       - **Default**: When not specified, defaults to analyzing last 7 days
            granularity: Time granularity (for trend mode), default "day" (hour/day)
            threshold: Hotness surge multiplier threshold (for viral mode), default 3.0
            time_window: Detection time window in hours (for viral mode), default 24
            lookahead_hours: Predict future hours (for predict mode), default 6
            confidence_threshold: Confidence threshold (for predict mode), default 0.7

        Returns:
            Trend analysis result dictionary

        Examples:
            - analyze_topic_trend_unified(topic="AI", analysis_type="trend", date_range={"start": "2025-10-18", "end": "2025-10-25"})
            - analyze_topic_trend_unified(topic="Tesla", analysis_type="lifecycle", date_range={"start": "2025-10-18", "end": "2025-10-25"})
            - analyze_topic_trend_unified(topic="Bitcoin", analysis_type="viral", threshold=3.0)
            - analyze_topic_trend_unified(topic="ChatGPT", analysis_type="predict", lookahead_hours=6)
        """
        try:
            # Parameter validation
            topic = validate_keyword(topic)

            if analysis_type not in ["trend", "lifecycle", "viral", "predict"]:
                raise InvalidParameterError(
                    f"Invalid analysis type: {analysis_type}",
                    suggestion="Supported types: trend, lifecycle, viral, predict"
                )

            # Call corresponding method based on analysis type
            if analysis_type == "trend":
                return self.get_topic_trend_analysis(
                    topic=topic,
                    date_range=date_range,
                    granularity=granularity
                )
            elif analysis_type == "lifecycle":
                return self.analyze_topic_lifecycle(
                    topic=topic,
                    date_range=date_range
                )
            elif analysis_type == "viral":
                # viral mode doesn't need topic parameter, uses general detection
                return self.detect_viral_topics(
                    threshold=threshold,
                    time_window=time_window
                )
            else:  # predict
                # predict mode doesn't need topic parameter, uses general prediction
                return self.predict_trending_topics(
                    lookahead_hours=lookahead_hours,
                    confidence_threshold=confidence_threshold
                )

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def get_topic_trend_analysis(
        self,
        topic: str,
        date_range: Optional[Dict[str, str]] = None,
        granularity: str = "day"
    ) -> Dict:
        """
        Hotness trend analysis - track hotness change trends for specific topics

        Args:
            topic: Topic keyword
            date_range: Date range (optional)
                       - **Format**: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
                       - **Default**: When not specified, defaults to analyzing last 7 days
            granularity: Time granularity, only supports day

        Returns:
            Trend analysis result dictionary

        Examples:
            User query examples:
            - "Help me analyze the hotness trend of 'AI' topic in the last week"
            - "View 'Bitcoin' hotness changes in the past week"
            - "See how 'iPhone' trend is in the last 7 days"
            - "Analyze 'Tesla' hotness trend in the last month"
            - "View 'ChatGPT' trend changes in December 2024"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> # Analyze 7-day trend
            >>> result = tools.get_topic_trend_analysis(
            ...     topic="AI",
            ...     date_range={"start": "2025-10-18", "end": "2025-10-25"},
            ...     granularity="day"
            ... )
            >>> # Analyze historical month trend
            >>> result = tools.get_topic_trend_analysis(
            ...     topic="Tesla",
            ...     date_range={"start": "2024-12-01", "end": "2024-12-31"},
            ...     granularity="day"
            ... )
            >>> print(result['trend_data'])
        """
        try:
            # Validate parameters
            topic = validate_keyword(topic)

            # Validate granularity parameter (only supports day)
            if granularity != "day":
                from ..utils.errors import InvalidParameterError
                raise InvalidParameterError(
                    f"Unsupported granularity parameter: {granularity}",
                    suggestion="Currently only supports 'day' granularity, because underlying data is aggregated by day"
                )

            # Process date range (default to last 7 days when not specified)
            if date_range:
                from ..utils.validators import validate_date_range
                date_range_tuple = validate_date_range(date_range)
                start_date, end_date = date_range_tuple
            else:
                # Default to last 7 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=6)

            # Collect trend data
            trend_data = []
            current_date = start_date

            while current_date <= end_date:
                try:
                    all_titles, _, _ = self.data_service.parser.read_all_titles_for_date(
                        date=current_date
                    )

                    # Count topic occurrence at this time point
                    count = 0
                    matched_titles = []

                    for _, titles in all_titles.items():
                        for title in titles.keys():
                            if topic.lower() in title.lower():
                                count += 1
                                matched_titles.append(title)

                    trend_data.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "count": count,
                        "sample_titles": matched_titles[:3]  # Only keep first 3 samples
                    })

                except DataNotFoundError:
                    trend_data.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "count": 0,
                        "sample_titles": []
                    })

                # Increment time by day
                current_date += timedelta(days=1)

            # Calculate trend indicators
            counts = [item["count"] for item in trend_data]
            total_days = (end_date - start_date).days + 1

            if len(counts) >= 2:
                # Calculate change rate
                first_non_zero = next((c for c in counts if c > 0), 0)
                last_count = counts[-1]

                if first_non_zero > 0:
                    change_rate = ((last_count - first_non_zero) / first_non_zero) * 100
                else:
                    change_rate = 0

                # Find peak time
                max_count = max(counts)
                peak_index = counts.index(max_count)
                peak_time = trend_data[peak_index]["date"]
            else:
                change_rate = 0
                peak_time = None
                max_count = 0

            return {
                "success": True,
                "topic": topic,
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d"),
                    "total_days": total_days
                },
                "granularity": granularity,
                "trend_data": trend_data,
                "statistics": {
                    "total_mentions": sum(counts),
                    "average_mentions": round(sum(counts) / len(counts), 2) if counts else 0,
                    "peak_count": max_count,
                    "peak_time": peak_time,
                    "change_rate": round(change_rate, 2)
                },
                "trend_direction": "rising" if change_rate > 10 else "falling" if change_rate < -10 else "stable"
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def compare_platforms(
        self,
        topic: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Platform comparison analysis - compare different platforms' attention to the same topic

        Args:
            topic: Topic keyword (optional, if not specified, compare overall activity)
            date_range: Date range, format: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}

        Returns:
            Platform comparison analysis results

        Examples:
            User query examples:
            - "Compare each platform's attention to 'AI' topic"
            - "See which platform pays more attention to tech news, CNN or BBC"
            - "Analyze today's hot topic distribution across platforms"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.compare_platforms(
            ...     topic="AI",
            ...     date_range={"start": "2025-10-01", "end": "2025-10-11"}
            ... )
            >>> print(result['platform_stats'])
        """
        try:
            # Parameter validation
            if topic:
                topic = validate_keyword(topic)
            date_range_tuple = validate_date_range(date_range)

            # Determine date range
            if date_range_tuple:
                start_date, end_date = date_range_tuple
            else:
                start_date = end_date = datetime.now()

            # Collect data from each platform
            platform_stats = defaultdict(lambda: {
                "total_news": 0,
                "topic_mentions": 0,
                "unique_titles": set(),
                "top_keywords": Counter()
            })

            # Iterate through date range
            current_date = start_date
            while current_date <= end_date:
                try:
                    all_titles, id_to_name, _ = self.data_service.parser.read_all_titles_for_date(
                        date=current_date
                    )

                    for platform_id, titles in all_titles.items():
                        platform_name = id_to_name.get(platform_id, platform_id)

                        for title in titles.keys():
                            platform_stats[platform_name]["total_news"] += 1
                            platform_stats[platform_name]["unique_titles"].add(title)

                            # If topic is specified, count news containing the topic
                            if topic and topic.lower() in title.lower():
                                platform_stats[platform_name]["topic_mentions"] += 1

                            # Extract keywords (simple word segmentation)
                            keywords = self._extract_keywords(title)
                            platform_stats[platform_name]["top_keywords"].update(keywords)

                except DataNotFoundError:
                    pass

                current_date += timedelta(days=1)

            # Convert to serializable format
            result_stats = {}
            for platform, stats in platform_stats.items():
                coverage_rate = 0
                if stats["total_news"] > 0:
                    coverage_rate = (stats["topic_mentions"] / stats["total_news"]) * 100

                result_stats[platform] = {
                    "total_news": stats["total_news"],
                    "topic_mentions": stats["topic_mentions"],
                    "unique_titles": len(stats["unique_titles"]),
                    "coverage_rate": round(coverage_rate, 2),
                    "top_keywords": [
                        {"keyword": k, "count": v}
                        for k, v in stats["top_keywords"].most_common(5)
                    ]
                }

            # Find unique hot topics for each platform
            unique_topics = self._find_unique_topics(platform_stats)

            return {
                "success": True,
                "topic": topic,
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                },
                "platform_stats": result_stats,
                "unique_topics": unique_topics,
                "total_platforms": len(result_stats)
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def analyze_keyword_cooccurrence(
        self,
        min_frequency: int = 3,
        top_n: int = 20
    ) -> Dict:
        """
        Keyword co-occurrence analysis - analyze which keywords often appear together

        Args:
            min_frequency: Minimum co-occurrence frequency
            top_n: Return TOP N keyword pairs

        Returns:
            Keyword co-occurrence analysis results

        Examples:
            User query examples:
            - "Analyze which keywords often appear together"
            - "See which words often appear with 'AI'"
            - "Find keyword associations in today's news"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.analyze_keyword_cooccurrence(
            ...     min_frequency=5,
            ...     top_n=15
            ... )
            >>> print(result['cooccurrence_pairs'])
        """
        try:
            # Parameter validation
            min_frequency = validate_limit(min_frequency, default=3, max_limit=100)
            top_n = validate_top_n(top_n, default=20)

            # Read today's data
            all_titles, _, _ = self.data_service.parser.read_all_titles_for_date()

            # Keyword co-occurrence statistics
            cooccurrence = Counter()
            keyword_titles = defaultdict(list)

            for platform_id, titles in all_titles.items():
                for title in titles.keys():
                    # Extract keywords
                    keywords = self._extract_keywords(title)

                    # Record titles where each keyword appears
                    for kw in keywords:
                        keyword_titles[kw].append(title)

                    # Calculate pairwise co-occurrence
                    if len(keywords) >= 2:
                        for i, kw1 in enumerate(keywords):
                            for kw2 in keywords[i+1:]:
                                # Unified sorting to avoid duplicates
                                pair = tuple(sorted([kw1, kw2]))
                                cooccurrence[pair] += 1

            # Filter low-frequency co-occurrences
            filtered_pairs = [
                (pair, count) for pair, count in cooccurrence.items()
                if count >= min_frequency
            ]

            # Sort and take TOP N
            top_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_n]

            # Build result
            result_pairs = []
            for (kw1, kw2), count in top_pairs:
                # Find title samples containing both keywords
                titles_with_both = [
                    title for title in keyword_titles[kw1]
                    if kw2 in self._extract_keywords(title)
                ]

                result_pairs.append({
                    "keyword1": kw1,
                    "keyword2": kw2,
                    "cooccurrence_count": count,
                    "sample_titles": titles_with_both[:3]
                })

            return {
                "success": True,
                "cooccurrence_pairs": result_pairs,
                "total_pairs": len(result_pairs),
                "min_frequency": min_frequency,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def analyze_sentiment(
        self,
        topic: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None,
        limit: int = 50,
        sort_by_weight: bool = True,
        include_url: bool = False
    ) -> Dict:
        """
        Sentiment analysis - generate structured prompts for AI sentiment analysis

        This tool collects news data and generates optimized AI prompts that you can send to AI for deep sentiment analysis.

        Args:
            topic: Topic keyword (optional), only analyze news containing this keyword
            platforms: Platform filter list (optional), e.g. ['cnn', 'bbc']
            date_range: Date range (optional), format: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
                       When not specified, defaults to querying today's data
            limit: Return news count limit, default 50, maximum 100
            sort_by_weight: Whether to sort by weight, default True (recommended)
            include_url: Whether to include URL links, default False (to save tokens)

        Returns:
            Structured result containing AI prompts and news data

        Examples:
            User query examples:
            - "Analyze sentiment tendency of today's news"
            - "See if 'Tesla' related news is positive or negative"
            - "Analyze sentiment attitude of each platform towards 'AI'"
            - "See if 'Tesla' related news is positive or negative, please select top 10 news from the week to analyze"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> # Analyze today's Tesla news, return top 10
            >>> result = tools.analyze_sentiment(
            ...     topic="Tesla",
            ...     limit=10
            ... )
            >>> # Analyze Tesla news within a week, return top 10 sorted by weight
            >>> result = tools.analyze_sentiment(
            ...     topic="Tesla",
            ...     date_range={"start": "2025-10-06", "end": "2025-10-13"},
            ...     limit=10
            ... )
            >>> print(result['ai_prompt'])  # Get generated prompt
        """
        try:
            # Parameter validation
            if topic:
                topic = validate_keyword(topic)
            platforms = validate_platforms(platforms)
            limit = validate_limit(limit, default=50)

            # Process date range
            if date_range:
                date_range_tuple = validate_date_range(date_range)
                start_date, end_date = date_range_tuple
            else:
                # Default today
                start_date = end_date = datetime.now()

            # Collect news data (supports multiple days)
            all_news_items = []
            current_date = start_date

            while current_date <= end_date:
                try:
                    all_titles, id_to_name, _ = self.data_service.parser.read_all_titles_for_date(
                        date=current_date,
                        platform_ids=platforms
                    )

                    # Collect news for this date
                    for platform_id, titles in all_titles.items():
                        platform_name = id_to_name.get(platform_id, platform_id)
                        for title, info in titles.items():
                            # If topic is specified, only collect titles containing the topic
                            if topic and topic.lower() not in title.lower():
                                continue

                            news_item = {
                                "platform": platform_name,
                                "title": title,
                                "ranks": info.get("ranks", []),
                                "count": len(info.get("ranks", [])),
                                "date": current_date.strftime("%Y-%m-%d")
                            }

                            # Conditionally add URL field
                            if include_url:
                                news_item["url"] = info.get("url", "")
                                news_item["mobileUrl"] = info.get("mobileUrl", "")

                            all_news_items.append(news_item)

                except DataNotFoundError:
                    # No data for this date, continue to next day
                    pass

                # Next day
                current_date += timedelta(days=1)

            if not all_news_items:
                time_desc = "today" if start_date == end_date else f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                raise DataNotFoundError(
                    f"No related news found ({time_desc})",
                    suggestion="Please try other topics, date ranges or platforms"
                )

            # Deduplicate (keep same title only once)
            unique_news = {}
            for item in all_news_items:
                key = f"{item['platform']}::{item['title']}"
                if key not in unique_news:
                    unique_news[key] = item
                else:
                    # Merge ranks (if same news appears on multiple days)
                    existing = unique_news[key]
                    existing["ranks"].extend(item["ranks"])
                    existing["count"] = len(existing["ranks"])

            deduplicated_news = list(unique_news.values())

            # Sort by weight (if enabled)
            if sort_by_weight:
                deduplicated_news.sort(
                    key=lambda x: calculate_news_weight(x),
                    reverse=True
                )

            # Limit return count
            selected_news = deduplicated_news[:limit]

            # Generate AI prompt
            ai_prompt = self._create_sentiment_analysis_prompt(
                news_data=selected_news,
                topic=topic
            )

            # Build time range description
            if start_date == end_date:
                time_range_desc = start_date.strftime("%Y-%m-%d")
            else:
                time_range_desc = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

            result = {
                "success": True,
                "method": "ai_prompt_generation",
                "summary": {
                    "total_found": len(deduplicated_news),
                    "returned_count": len(selected_news),
                    "requested_limit": limit,
                    "duplicates_removed": len(all_news_items) - len(deduplicated_news),
                    "topic": topic,
                    "time_range": time_range_desc,
                    "platforms": list(set(item["platform"] for item in selected_news)),
                    "sorted_by_weight": sort_by_weight
                },
                "ai_prompt": ai_prompt,
                "news_sample": selected_news,
                "usage_note": "Please send the content of the ai_prompt field to AI for sentiment analysis"
            }

            # If return count is less than requested count, add note
            if len(selected_news) < limit and len(deduplicated_news) >= limit:
                result["note"] = "Return count is less than requested because of deduplication logic (same title on different platforms only kept once)"
            elif len(deduplicated_news) < limit:
                result["note"] = f"Only found {len(deduplicated_news)} matching news items in the specified time range"

            return result

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def _create_sentiment_analysis_prompt(
        self,
        news_data: List[Dict],
        topic: Optional[str]
    ) -> str:
        """
        Create AI prompt for sentiment analysis

        Args:
            news_data: News data list (already sorted and limited)
            topic: Topic keyword

        Returns:
            Formatted AI prompt
        """
        # Group by platform
        platform_news = defaultdict(list)
        for item in news_data:
            platform_news[item["platform"]].append({
                "title": item["title"],
                "date": item.get("date", "")
            })

        # Build prompt
        prompt_parts = []

        # 1. Task description
        if topic:
            prompt_parts.append(f"Please analyze the sentiment tendency of the following news titles about '{topic}'.")
        else:
            prompt_parts.append("Please analyze the sentiment tendency of the following news titles.")

        prompt_parts.append("")
        prompt_parts.append("Analysis requirements:")
        prompt_parts.append("1. Identify sentiment tendency of each news item (positive/negative/neutral)")
        prompt_parts.append("2. Count and calculate percentage for each sentiment category")
        prompt_parts.append("3. Analyze sentiment differences across platforms")
        prompt_parts.append("4. Summarize overall sentiment trend")
        prompt_parts.append("5. List typical positive and negative news samples")
        prompt_parts.append("")

        # 2. Data overview
        prompt_parts.append(f"Data Overview:")
        prompt_parts.append(f"- Total news count: {len(news_data)}")
        prompt_parts.append(f"- Platforms covered: {len(platform_news)}")

        # Time range
        dates = set(item.get("date", "") for item in news_data if item.get("date"))
        if dates:
            date_list = sorted(dates)
            if len(date_list) == 1:
                prompt_parts.append(f"- Time range: {date_list[0]}")
            else:
                prompt_parts.append(f"- Time range: {date_list[0]} to {date_list[-1]}")

        prompt_parts.append("")

        # 3. Display news by platform
        prompt_parts.append("News List (grouped by platform, sorted by importance):")
        prompt_parts.append("")

        for platform, items in sorted(platform_news.items()):
            prompt_parts.append(f"[{platform}] ({len(items)} items)")
            for i, item in enumerate(items, 1):
                title = item["title"]
                date_str = f" [{item['date']}]" if item.get("date") else ""
                prompt_parts.append(f"{i}. {title}{date_str}")
            prompt_parts.append("")

        # 4. Output format instructions
        prompt_parts.append("Please output analysis results in the following format:")
        prompt_parts.append("")
        prompt_parts.append("## Sentiment Distribution Statistics")
        prompt_parts.append("- Positive: XX items (XX%)")
        prompt_parts.append("- Negative: XX items (XX%)")
        prompt_parts.append("- Neutral: XX items (XX%)")
        prompt_parts.append("")
        prompt_parts.append("## Platform Sentiment Comparison")
        prompt_parts.append("[Sentiment tendency differences across platforms]")
        prompt_parts.append("")
        prompt_parts.append("## Overall Sentiment Trend")
        prompt_parts.append("[Overall analysis and key findings]")
        prompt_parts.append("")
        prompt_parts.append("## Typical Samples")
        prompt_parts.append("Positive news samples:")
        prompt_parts.append("[List 3-5 items]")
        prompt_parts.append("")
        prompt_parts.append("Negative news samples:")
        prompt_parts.append("[List 3-5 items]")

        return "\n".join(prompt_parts)

    def find_similar_news(
        self,
        reference_title: str,
        threshold: float = 0.6,
        limit: int = 50,
        include_url: bool = False
    ) -> Dict:
        """
        Similar news search - find related news based on title similarity

        Args:
            reference_title: Reference title
            threshold: Similarity threshold (between 0-1)
            limit: Return count limit, default 50
            include_url: Whether to include URL links, default False (to save tokens)

        Returns:
            Similar news list

        Examples:
            User query examples:
            - "Find news similar to 'Tesla price cut'"
            - "Search for similar reports about iPhone release"
            - "See if there are similar reports to this news"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.find_similar_news(
            ...     reference_title="Tesla announces price cut",
            ...     threshold=0.6,
            ...     limit=10
            ... )
            >>> print(result['similar_news'])
        """
        try:
            # Parameter validation
            reference_title = validate_keyword(reference_title)

            if not 0 <= threshold <= 1:
                raise InvalidParameterError(
                    "threshold must be between 0 and 1",
                    suggestion="Recommended values: 0.5-0.8"
                )

            limit = validate_limit(limit, default=50)

            # Read data
            all_titles, id_to_name, _ = self.data_service.parser.read_all_titles_for_date()

            # Calculate similarity
            similar_items = []

            for platform_id, titles in all_titles.items():
                platform_name = id_to_name.get(platform_id, platform_id)

                for title, info in titles.items():
                    if title == reference_title:
                        continue

                    # Calculate similarity
                    similarity = self._calculate_similarity(reference_title, title)

                    if similarity >= threshold:
                        news_item = {
                            "title": title,
                            "platform": platform_id,
                            "platform_name": platform_name,
                            "similarity": round(similarity, 3),
                            "rank": info["ranks"][0] if info["ranks"] else 0
                        }

                        # Conditionally add URL field
                        if include_url:
                            news_item["url"] = info.get("url", "")

                        similar_items.append(news_item)

            # Sort by similarity
            similar_items.sort(key=lambda x: x["similarity"], reverse=True)

            # Limit count
            result_items = similar_items[:limit]

            if not result_items:
                raise DataNotFoundError(
                    f"No news found with similarity exceeding {threshold}",
                    suggestion="Please lower similarity threshold or try other titles"
                )

            result = {
                "success": True,
                "summary": {
                    "total_found": len(similar_items),
                    "returned_count": len(result_items),
                    "requested_limit": limit,
                    "threshold": threshold,
                    "reference_title": reference_title
                },
                "similar_news": result_items
            }

            if len(similar_items) < limit:
                result["note"] = f"With similarity threshold {threshold}, only found {len(similar_items)} similar news items"

            return result

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def search_by_entity(
        self,
        entity: str,
        entity_type: Optional[str] = None,
        limit: int = 50,
        sort_by_weight: bool = True
    ) -> Dict:
        """
        Entity recognition search - search for news containing specific person/location/organization

        Args:
            entity: Entity name
            entity_type: Entity type (person/location/organization), optional
            limit: Return count limit, default 50, maximum 200
            sort_by_weight: Whether to sort by weight, default True

        Returns:
            Entity-related news list

        Examples:
            User query examples:
            - "Search for news related to Musk"
            - "Find reports about Tesla company, return top 20"
            - "See what news there is about Washington"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.search_by_entity(
            ...     entity="Musk",
            ...     entity_type="person",
            ...     limit=20
            ... )
            >>> print(result['related_news'])
        """
        try:
            # Parameter validation
            entity = validate_keyword(entity)
            limit = validate_limit(limit, default=50)

            if entity_type and entity_type not in ["person", "location", "organization"]:
                raise InvalidParameterError(
                    f"Invalid entity type: {entity_type}",
                    suggestion="Supported types: person, location, organization"
                )

            # Read data
            all_titles, id_to_name, _ = self.data_service.parser.read_all_titles_for_date()

            # Search for news containing entity
            related_news = []
            entity_context = Counter()  # Count words around entity

            for platform_id, titles in all_titles.items():
                platform_name = id_to_name.get(platform_id, platform_id)

                for title, info in titles.items():
                    if entity in title:
                        url = info.get("url", "")
                        mobile_url = info.get("mobileUrl", "")
                        ranks = info.get("ranks", [])
                        count = len(ranks)

                        related_news.append({
                            "title": title,
                            "platform": platform_id,
                            "platform_name": platform_name,
                            "url": url,
                            "mobileUrl": mobile_url,
                            "ranks": ranks,
                            "count": count,
                            "rank": ranks[0] if ranks else 999
                        })

                        # Extract keywords around entity
                        keywords = self._extract_keywords(title)
                        entity_context.update(keywords)

            if not related_news:
                raise DataNotFoundError(
                    f"No news found containing entity '{entity}'",
                    suggestion="Please try other entity names"
                )

            # Remove entity itself
            if entity in entity_context:
                del entity_context[entity]

            # Sort by weight (if enabled)
            if sort_by_weight:
                related_news.sort(
                    key=lambda x: calculate_news_weight(x),
                    reverse=True
                )
            else:
                # Sort by rank
                related_news.sort(key=lambda x: x["rank"])

            # Limit return count
            result_news = related_news[:limit]

            return {
                "success": True,
                "entity": entity,
                "entity_type": entity_type or "auto",
                "related_news": result_news,
                "total_found": len(related_news),
                "returned_count": len(result_news),
                "sorted_by_weight": sort_by_weight,
                "related_keywords": [
                    {"keyword": k, "count": v}
                    for k, v in entity_context.most_common(10)
                ]
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def generate_summary_report(
        self,
        report_type: str = "daily",
        date_range: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Daily/Weekly summary generator - automatically generates hot topics summary report

        Args:
            report_type: Report type (daily/weekly)
            date_range: Custom date range (optional)

        Returns:
            Markdown format summary report

        Examples:
            User query examples:
            - "Generate today's news summary report"
            - "Give me a summary of this week's hot topics"
            - "Generate news analysis report for the past 7 days"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.generate_summary_report(
            ...     report_type="daily"
            ... )
            >>> print(result['markdown_report'])
        """
        try:
            # Parameter validation
            if report_type not in ["daily", "weekly"]:
                raise InvalidParameterError(
                    f"Invalid report type: {report_type}",
                    suggestion="Supported types: daily, weekly"
                )

            # Determine date range
            if date_range:
                date_range_tuple = validate_date_range(date_range)
                start_date, end_date = date_range_tuple
            else:
                if report_type == "daily":
                    start_date = end_date = datetime.now()
                else:  # weekly
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=6)

            # Collect data
            all_keywords = Counter()
            all_platforms_news = defaultdict(int)
            all_titles_list = []

            current_date = start_date
            while current_date <= end_date:
                try:
                    all_titles, id_to_name, _ = self.data_service.parser.read_all_titles_for_date(
                        date=current_date
                    )

                    for platform_id, titles in all_titles.items():
                        platform_name = id_to_name.get(platform_id, platform_id)
                        all_platforms_news[platform_name] += len(titles)

                        for title in titles.keys():
                            all_titles_list.append({
                                "title": title,
                                "platform": platform_name,
                                "date": current_date.strftime("%Y-%m-%d")
                            })

                            # Extract keywords
                            keywords = self._extract_keywords(title)
                            all_keywords.update(keywords)

                except DataNotFoundError:
                    pass

                current_date += timedelta(days=1)

            # Generate report
            report_title = f"{'Daily' if report_type == 'daily' else 'Weekly'} News Hot Topics Summary"
            date_str = f"{start_date.strftime('%Y-%m-%d')}" if report_type == "daily" else f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

            # Build Markdown report
            markdown = f"""# {report_title}

**Report Date**: {date_str}
**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Data Overview

- **Total News Count**: {len(all_titles_list)}
- **Platforms Covered**: {len(all_platforms_news)}
- **Hot Keywords Count**: {len(all_keywords)}

## 🔥 TOP 10 Hot Topics

"""

            # Add TOP 10 keywords
            for i, (keyword, count) in enumerate(all_keywords.most_common(10), 1):
                markdown += f"{i}. **{keyword}** - appeared {count} times\n"

            # Platform analysis
            markdown += "\n## 📱 Platform Activity\n\n"
            sorted_platforms = sorted(all_platforms_news.items(), key=lambda x: x[1], reverse=True)

            for platform, count in sorted_platforms:
                markdown += f"- **{platform}**: {count} news items\n"

            # Trend changes (if weekly report)
            if report_type == "weekly":
                markdown += "\n## 📈 Trend Analysis\n\n"
                markdown += "Topics with sustained hotness this week (sample data):\n\n"

                # Simple trend analysis
                top_keywords = [kw for kw, _ in all_keywords.most_common(5)]
                for keyword in top_keywords:
                    markdown += f"- **{keyword}**: Continues to be hot\n"

            # Add sample news (selected by weight, ensuring determinism)
            markdown += "\n## 📰 Featured News Samples\n\n"

            # Deterministic selection: sort by title weight, take top 5
            # This ensures same input always returns same result
            if all_titles_list:
                # Calculate weight score for each news item (based on keyword occurrence count)
                news_with_scores = []
                for news in all_titles_list:
                    # Simple weight: count occurrences of TOP keywords
                    score = 0
                    title_lower = news['title'].lower()
                    for keyword, count in all_keywords.most_common(10):
                        if keyword.lower() in title_lower:
                            score += count
                    news_with_scores.append((news, score))

                # Sort by weight descending, if same weight then by title alphabetical (ensuring determinism)
                news_with_scores.sort(key=lambda x: (-x[1], x[0]['title']))

                # Take top 5
                sample_news = [item[0] for item in news_with_scores[:5]]

                for news in sample_news:
                    markdown += f"- [{news['platform']}] {news['title']}\n"

            markdown += "\n---\n\n*This report is automatically generated by TrendRadar MCP*\n"

            return {
                "success": True,
                "report_type": report_type,
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                },
                "markdown_report": markdown,
                "statistics": {
                    "total_news": len(all_titles_list),
                    "platforms_count": len(all_platforms_news),
                    "keywords_count": len(all_keywords),
                    "top_keyword": all_keywords.most_common(1)[0] if all_keywords else None
                }
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def get_platform_activity_stats(
        self,
        date_range: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Platform activity statistics - statistics on each platform's publishing frequency and active time periods

        Args:
            date_range: Date range (optional)

        Returns:
            Platform activity statistics results

        Examples:
            User query examples:
            - "Statistics on each platform's activity today"
            - "See which platform updates most frequently"
            - "Analyze publishing time patterns for each platform"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.get_platform_activity_stats(
            ...     date_range={"start": "2025-10-01", "end": "2025-10-11"}
            ... )
            >>> print(result['platform_activity'])
        """
        try:
            # Parameter validation
            date_range_tuple = validate_date_range(date_range)

            # Determine date range
            if date_range_tuple:
                start_date, end_date = date_range_tuple
            else:
                start_date = end_date = datetime.now()

            # Statistics on each platform's activity
            platform_activity = defaultdict(lambda: {
                "total_updates": 0,
                "days_active": set(),
                "news_count": 0,
                "hourly_distribution": Counter()
            })

            # Iterate through date range
            current_date = start_date
            while current_date <= end_date:
                try:
                    all_titles, id_to_name, timestamps = self.data_service.parser.read_all_titles_for_date(
                        date=current_date
                    )

                    for platform_id, titles in all_titles.items():
                        platform_name = id_to_name.get(platform_id, platform_id)

                        platform_activity[platform_name]["news_count"] += len(titles)
                        platform_activity[platform_name]["days_active"].add(current_date.strftime("%Y-%m-%d"))

                        # Count update frequency (based on file count)
                        platform_activity[platform_name]["total_updates"] += len(timestamps)

                        # Count time distribution (based on time in filename)
                        for filename in timestamps.keys():
                            # Parse hour from filename (format: HHMM.txt)
                            match = re.match(r'(\d{2})(\d{2})\.txt', filename)
                            if match:
                                hour = int(match.group(1))
                                platform_activity[platform_name]["hourly_distribution"][hour] += 1

                except DataNotFoundError:
                    pass

                current_date += timedelta(days=1)

            # Convert to serializable format
            result_activity = {}
            for platform, stats in platform_activity.items():
                days_count = len(stats["days_active"])
                avg_news_per_day = stats["news_count"] / days_count if days_count > 0 else 0

                # Find most active time period
                most_active_hours = stats["hourly_distribution"].most_common(3)

                result_activity[platform] = {
                    "total_updates": stats["total_updates"],
                    "news_count": stats["news_count"],
                    "days_active": days_count,
                    "avg_news_per_day": round(avg_news_per_day, 2),
                    "most_active_hours": [
                        {"hour": f"{hour:02d}:00", "count": count}
                        for hour, count in most_active_hours
                    ],
                    "activity_score": round(stats["news_count"] / max(days_count, 1), 2)
                }

            # Sort by activity
            sorted_platforms = sorted(
                result_activity.items(),
                key=lambda x: x[1]["activity_score"],
                reverse=True
            )

            return {
                "success": True,
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                },
                "platform_activity": dict(sorted_platforms),
                "most_active_platform": sorted_platforms[0][0] if sorted_platforms else None,
                "total_platforms": len(result_activity)
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def analyze_topic_lifecycle(
        self,
        topic: str,
        date_range: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Topic lifecycle analysis - track complete cycle of topic from appearance to disappearance

        Args:
            topic: Topic keyword
            date_range: Date range (optional)
                       - **Format**: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
                       - **Default**: When not specified, defaults to analyzing last 7 days

        Returns:
            Topic lifecycle analysis results

        Examples:
            User query examples:
            - "Analyze the lifecycle of 'AI' topic"
            - "See if 'iPhone' topic is a flash in the pan or sustained hot topic"
            - "Track hotness changes of 'Bitcoin' topic"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.analyze_topic_lifecycle(
            ...     topic="AI",
            ...     date_range={"start": "2025-10-18", "end": "2025-10-25"}
            ... )
            >>> print(result['lifecycle_stage'])
        """
        try:
            # Parameter validation
            topic = validate_keyword(topic)

            # Process date range (default to last 7 days when not specified)
            if date_range:
                from ..utils.validators import validate_date_range
                date_range_tuple = validate_date_range(date_range)
                start_date, end_date = date_range_tuple
            else:
                # Default to last 7 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=6)

            # Collect topic historical data
            lifecycle_data = []
            current_date = start_date
            while current_date <= end_date:
                try:
                    all_titles, _, _ = self.data_service.parser.read_all_titles_for_date(
                        date=current_date
                    )

                    # Count topic occurrence for this day
                    count = 0
                    for _, titles in all_titles.items():
                        for title in titles.keys():
                            if topic.lower() in title.lower():
                                count += 1

                    lifecycle_data.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "count": count
                    })

                except DataNotFoundError:
                    lifecycle_data.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "count": 0
                    })

                current_date += timedelta(days=1)

            # Calculate analysis days
            total_days = (end_date - start_date).days + 1

            # Analyze lifecycle stage
            counts = [item["count"] for item in lifecycle_data]

            if not any(counts):
                time_desc = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                raise DataNotFoundError(
                    f"Topic '{topic}' not found in {time_desc}",
                    suggestion="Please try other topics or expand time range"
                )

            # Find first and last appearance
            first_appearance = next((item["date"] for item in lifecycle_data if item["count"] > 0), None)
            last_appearance = next((item["date"] for item in reversed(lifecycle_data) if item["count"] > 0), None)

            # Calculate peak
            max_count = max(counts)
            peak_index = counts.index(max_count)
            peak_date = lifecycle_data[peak_index]["date"]

            # Calculate average and standard deviation (simple implementation)
            non_zero_counts = [c for c in counts if c > 0]
            avg_count = sum(non_zero_counts) / len(non_zero_counts) if non_zero_counts else 0

            # Determine lifecycle stage
            recent_counts = counts[-3:]  # Last 3 days
            early_counts = counts[:3]    # First 3 days

            if sum(recent_counts) > sum(early_counts):
                lifecycle_stage = "rising"
            elif sum(recent_counts) < sum(early_counts) * 0.5:
                lifecycle_stage = "declining"
            elif max_count in recent_counts:
                lifecycle_stage = "surging"
            else:
                lifecycle_stage = "stable"

            # Classification: flash in the pan vs sustained hot topic
            active_days = sum(1 for c in counts if c > 0)

            if active_days <= 2 and max_count > avg_count * 2:
                topic_type = "flash in the pan"
            elif active_days >= total_days * 0.6:
                topic_type = "sustained hot topic"
            else:
                topic_type = "periodic hot topic"

            return {
                "success": True,
                "topic": topic,
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d"),
                    "total_days": total_days
                },
                "lifecycle_data": lifecycle_data,
                "analysis": {
                    "first_appearance": first_appearance,
                    "last_appearance": last_appearance,
                    "peak_date": peak_date,
                    "peak_count": max_count,
                    "active_days": active_days,
                    "avg_daily_mentions": round(avg_count, 2),
                    "lifecycle_stage": lifecycle_stage,
                    "topic_type": topic_type
                }
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def detect_viral_topics(
        self,
        threshold: float = 3.0,
        time_window: int = 24
    ) -> Dict:
        """
        Abnormal hotness detection - automatically identify suddenly trending topics

        Args:
            threshold: Hotness surge multiplier threshold
            time_window: Detection time window (hours)

        Returns:
            List of viral topics

        Examples:
            User query examples:
            - "Detect which topics suddenly went viral today"
            - "See if there are any news with abnormal hotness"
            - "Alert on possible major events"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.detect_viral_topics(
            ...     threshold=3.0,
            ...     time_window=24
            ... )
            >>> print(result['viral_topics'])
        """
        try:
            # Parameter validation
            if threshold < 1.0:
                raise InvalidParameterError(
                    "threshold must be greater than or equal to 1.0",
                    suggestion="Recommended value: 2.0-5.0"
                )

            time_window = validate_limit(time_window, default=24, max_limit=72)

            # Read current and previous data
            current_all_titles, _, _ = self.data_service.parser.read_all_titles_for_date()

            # Read yesterday's data as baseline
            yesterday = datetime.now() - timedelta(days=1)
            try:
                previous_all_titles, _, _ = self.data_service.parser.read_all_titles_for_date(
                    date=yesterday
                )
            except DataNotFoundError:
                previous_all_titles = {}

            # Count current keyword frequency
            current_keywords = Counter()
            current_keyword_titles = defaultdict(list)

            for _, titles in current_all_titles.items():
                for title in titles.keys():
                    keywords = self._extract_keywords(title)
                    current_keywords.update(keywords)

                    for kw in keywords:
                        current_keyword_titles[kw].append(title)

            # Count previous keyword frequency
            previous_keywords = Counter()

            for _, titles in previous_all_titles.items():
                for title in titles.keys():
                    keywords = self._extract_keywords(title)
                    previous_keywords.update(keywords)

            # Detect abnormal hotness
            viral_topics = []

            for keyword, current_count in current_keywords.items():
                previous_count = previous_keywords.get(keyword, 0)

                # Calculate growth rate
                if previous_count == 0:
                    # Newly appeared topic
                    if current_count >= 5:  # At least 5 occurrences to be considered viral
                        growth_rate = float('inf')
                        is_viral = True
                    else:
                        continue
                else:
                    growth_rate = current_count / previous_count
                    is_viral = growth_rate >= threshold

                if is_viral:
                    viral_topics.append({
                        "keyword": keyword,
                        "current_count": current_count,
                        "previous_count": previous_count,
                        "growth_rate": round(growth_rate, 2) if growth_rate != float('inf') else "new topic",
                        "sample_titles": current_keyword_titles[keyword][:3],
                        "alert_level": "high" if growth_rate > threshold * 2 else "medium"
                    })

            # Sort by growth rate
            viral_topics.sort(
                key=lambda x: x["current_count"] if x["growth_rate"] == "new topic" else x["growth_rate"],
                reverse=True
            )

            if not viral_topics:
                return {
                    "success": True,
                    "viral_topics": [],
                    "total_detected": 0,
                    "message": f"No topics detected with hotness growth exceeding {threshold}x"
                }

            return {
                "success": True,
                "viral_topics": viral_topics,
                "total_detected": len(viral_topics),
                "threshold": threshold,
                "time_window": time_window,
                "detection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    def predict_trending_topics(
        self,
        lookahead_hours: int = 6,
        confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Topic prediction - predict possible future hot topics based on historical data

        Args:
            lookahead_hours: Predict how many hours into the future
            confidence_threshold: Confidence threshold

        Returns:
            List of predicted potential topics

        Examples:
            User query examples:
            - "Predict possible hot topics in the next 6 hours"
            - "What topics might become trending"
            - "Early detection of potential topics"

            Code call examples:
            >>> tools = AnalyticsTools()
            >>> result = tools.predict_trending_topics(
            ...     lookahead_hours=6,
            ...     confidence_threshold=0.7
            ... )
            >>> print(result['predicted_topics'])
        """
        try:
            # Parameter validation
            lookahead_hours = validate_limit(lookahead_hours, default=6, max_limit=48)

            if not 0 <= confidence_threshold <= 1:
                raise InvalidParameterError(
                    "confidence_threshold must be between 0 and 1",
                    suggestion="Recommended value: 0.6-0.8"
                )

            # Collect data from last 3 days for prediction
            keyword_trends = defaultdict(list)

            for days_ago in range(3, 0, -1):
                date = datetime.now() - timedelta(days=days_ago)

                try:
                    all_titles, _, _ = self.data_service.parser.read_all_titles_for_date(
                        date=date
                    )

                    # Count keywords
                    keywords_count = Counter()
                    for _, titles in all_titles.items():
                        for title in titles.keys():
                            keywords = self._extract_keywords(title)
                            keywords_count.update(keywords)

                    # Record historical data for each keyword
                    for keyword, count in keywords_count.items():
                        keyword_trends[keyword].append(count)

                except DataNotFoundError:
                    pass

            # Add today's data
            try:
                all_titles, _, _ = self.data_service.parser.read_all_titles_for_date()

                keywords_count = Counter()
                keyword_titles = defaultdict(list)

                for _, titles in all_titles.items():
                    for title in titles.keys():
                        keywords = self._extract_keywords(title)
                        keywords_count.update(keywords)

                        for kw in keywords:
                            keyword_titles[kw].append(title)

                for keyword, count in keywords_count.items():
                    keyword_trends[keyword].append(count)

            except DataNotFoundError:
                raise DataNotFoundError(
                    "Today's data not found",
                    suggestion="Please wait for crawler task to complete"
                )

            # Predict potential topics
            predicted_topics = []

            for keyword, trend_data in keyword_trends.items():
                if len(trend_data) < 2:
                    continue

                # Simple linear trend prediction
                # Calculate growth rate
                recent_value = trend_data[-1]
                previous_value = trend_data[-2] if len(trend_data) >= 2 else 0

                if previous_value == 0:
                    if recent_value >= 3:
                        growth_rate = 1.0
                    else:
                        continue
                else:
                    growth_rate = (recent_value - previous_value) / previous_value

                # Determine if it's an upward trend
                if growth_rate > 0.3:  # Growth over 30%
                    # Calculate confidence (based on trend stability)
                    if len(trend_data) >= 3:
                        # Check if continuously growing
                        is_consistent = all(
                            trend_data[i] <= trend_data[i+1]
                            for i in range(len(trend_data)-1)
                        )
                        confidence = 0.9 if is_consistent else 0.7
                    else:
                        confidence = 0.6

                    if confidence >= confidence_threshold:
                        predicted_topics.append({
                            "keyword": keyword,
                            "current_count": recent_value,
                            "growth_rate": round(growth_rate * 100, 2),
                            "confidence": round(confidence, 2),
                            "trend_data": trend_data,
                            "prediction": "Upward trend, likely to become hot topic",
                            "sample_titles": keyword_titles.get(keyword, [])[:3]
                        })

            # Sort by confidence and growth rate
            predicted_topics.sort(
                key=lambda x: (x["confidence"], x["growth_rate"]),
                reverse=True
            )

            return {
                "success": True,
                "predicted_topics": predicted_topics[:20],  # Return TOP 20
                "total_predicted": len(predicted_topics),
                "lookahead_hours": lookahead_hours,
                "confidence_threshold": confidence_threshold,
                "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Predictions based on historical trends, actual results may vary"
            }

        except MCPError as e:
            return {
                "success": False,
                "error": e.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }

    # ==================== Helper Methods ====================

    def _extract_keywords(self, title: str, min_length: int = 2) -> List[str]:
        """
        Extract keywords from title (simple implementation)

        Args:
            title: Title text
            min_length: Minimum keyword length

        Returns:
            Keyword list
        """
        # Remove URLs and special characters
        title = re.sub(r'http[s]?://\S+', '', title)
        title = re.sub(r'[^\w\s]', ' ', title)

        # Simple word segmentation (by spaces and common delimiters)
        words = re.split(r'[\s，。！？、]+', title)

        # Filter stopwords and short words
        stopwords = {
            'a', 'an', 'the', 'and', 'but', 'or', 'of', 'to', 'in', 'for', 'on',
            'with', 'as', 'at', 'by', 'from', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now'
        }

        keywords = [
            word.strip() for word in words
            if word.strip() and len(word.strip()) >= min_length and word.strip() not in stopwords
        ]

        return keywords

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts

        Args:
            text1: Text 1
            text2: Text 2

        Returns:
            Similarity score (between 0-1)
        """
        # Use SequenceMatcher to calculate similarity
        return SequenceMatcher(None, text1, text2).ratio()

    def _find_unique_topics(self, platform_stats: Dict) -> Dict[str, List[str]]:
        """
        Find unique hot topics for each platform

        Args:
            platform_stats: Platform statistics data

        Returns:
            Dictionary of unique topics for each platform
        """
        unique_topics = {}

        # Get TOP keywords for each platform
        platform_keywords = {}
        for platform, stats in platform_stats.items():
            top_keywords = set([kw for kw, _ in stats["top_keywords"].most_common(10)])
            platform_keywords[platform] = top_keywords

        # Find unique keywords
        for platform, keywords in platform_keywords.items():
            # Find all keywords from other platforms
            other_keywords = set()
            for other_platform, other_kws in platform_keywords.items():
                if other_platform != platform:
                    other_keywords.update(other_kws)

            # Find unique ones
            unique = keywords - other_keywords
            if unique:
                unique_topics[platform] = list(unique)[:5]  # Maximum 5

        return unique_topics
