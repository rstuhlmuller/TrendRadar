"""
Intelligent News Search Tools

Provides advanced search functionality including fuzzy search, link queries, historical related news search, etc.
"""

import re
from collections import Counter
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from ..services.data_service import DataService
from ..utils.validators import validate_keyword, validate_limit
from ..utils.errors import MCPError, InvalidParameterError, DataNotFoundError


class SearchTools:
    """Intelligent News Search Tools Class"""

    def __init__(self, project_root: str = None):
        """
        Initialize intelligent search tools

        Args:
            project_root: Project root directory
        """
        self.data_service = DataService(project_root)
        # English stopwords list
        self.stopwords = {
            'a', 'an', 'the', 'and', 'but', 'or', 'of', 'to', 'in', 'for', 'on',
            'with', 'as', 'at', 'by', 'from', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now'
        }

    def search_news_unified(
        self,
        query: str,
        search_mode: str = "keyword",
        date_range: Optional[Dict[str, str]] = None,
        platforms: Optional[List[str]] = None,
        limit: int = 50,
        sort_by: str = "relevance",
        threshold: float = 0.6,
        include_url: bool = False
    ) -> Dict:
        """
        Unified news search tool - integrates multiple search modes

        Args:
            query: Query content (required) - keyword, content fragment or entity name
            search_mode: Search mode, optional values:
                - "keyword": Exact keyword matching (default)
                - "fuzzy": Fuzzy content matching (uses similarity algorithm)
                - "entity": Entity name search (automatically sorted by weight)
            date_range: Date range (optional)
                       - **Format**: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
                       - **Example**: {"start": "2025-01-01", "end": "2025-01-07"}
                       - **Default**: When not specified, defaults to querying today
                       - **Note**: start and end can be the same (represents single day query)
            platforms: Platform filter list, e.g. ['cnn', 'bbc']
            limit: Return count limit, default 50
            sort_by: Sort method, optional values:
                - "relevance": Sort by relevance (default)
                - "weight": Sort by news weight
                - "date": Sort by date
            threshold: Similarity threshold (only effective for fuzzy mode), between 0-1, default 0.6
            include_url: Whether to include URL links, default False (to save tokens)

        Returns:
            Search result dictionary, containing matched news list

        Examples:
            - search_news_unified(query="AI", search_mode="keyword")
            - search_news_unified(query="Tesla price cut", search_mode="fuzzy", threshold=0.4)
            - search_news_unified(query="Musk", search_mode="entity", limit=20)
            - search_news_unified(query="iPhone 16", date_range={"start": "2025-01-01", "end": "2025-01-07"})
        """
        try:
            # Parameter validation
            query = validate_keyword(query)

            if search_mode not in ["keyword", "fuzzy", "entity"]:
                raise InvalidParameterError(
                    f"Invalid search mode: {search_mode}",
                    suggestion="Supported modes: keyword, fuzzy, entity"
                )

            if sort_by not in ["relevance", "weight", "date"]:
                raise InvalidParameterError(
                    f"Invalid sort method: {sort_by}",
                    suggestion="Supported sorts: relevance, weight, date"
                )

            limit = validate_limit(limit, default=50)
            threshold = max(0.0, min(1.0, threshold))

            # Process date range
            if date_range:
                from ..utils.validators import validate_date_range
                date_range_tuple = validate_date_range(date_range)
                start_date, end_date = date_range_tuple
            else:
                # When date not specified, use latest available data date (not datetime.now())
                earliest, latest = self.data_service.get_available_date_range()

                if latest is None:
                    # No available data
                    return {
                        "success": False,
                        "error": {
                            "code": "NO_DATA_AVAILABLE",
                            "message": "No available news data in output directory",
                            "suggestion": "Please run crawler to generate data first, or check output directory"
                        }
                    }

                # Use latest available date
                start_date = end_date = latest

            # Collect all matching news
            all_matches = []
            current_date = start_date

            while current_date <= end_date:
                try:
                    all_titles, id_to_name, timestamps = self.data_service.parser.read_all_titles_for_date(
                        date=current_date,
                        platform_ids=platforms
                    )

                    # Execute different search logic based on search mode
                    if search_mode == "keyword":
                        matches = self._search_by_keyword_mode(
                            query, all_titles, id_to_name, current_date, include_url
                        )
                    elif search_mode == "fuzzy":
                        matches = self._search_by_fuzzy_mode(
                            query, all_titles, id_to_name, current_date, threshold, include_url
                        )
                    else:  # entity
                        matches = self._search_by_entity_mode(
                            query, all_titles, id_to_name, current_date, include_url
                        )

                    all_matches.extend(matches)

                except DataNotFoundError:
                    # No data for this date, continue to next day
                    pass

                current_date += timedelta(days=1)

            if not all_matches:
                # Get available date range for error message
                earliest, latest = self.data_service.get_available_date_range()

                # Determine time range description
                if start_date.date() == datetime.now().date() and start_date == end_date:
                    time_desc = "today"
                elif start_date == end_date:
                    time_desc = start_date.strftime("%Y-%m-%d")
                else:
                    time_desc = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

                # Build error message
                if earliest and latest:
                    available_desc = f"{earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"
                    message = f"No matching news found (query range: {time_desc}, available data: {available_desc})"
                else:
                    message = f"No matching news found ({time_desc})"

                result = {
                    "success": True,
                    "results": [],
                    "total": 0,
                    "query": query,
                    "search_mode": search_mode,
                    "time_range": time_desc,
                    "message": message
                }
                return result

            # Unified sorting logic
            if sort_by == "relevance":
                all_matches.sort(key=lambda x: x.get("similarity_score", 1.0), reverse=True)
            elif sort_by == "weight":
                from .analytics import calculate_news_weight
                all_matches.sort(key=lambda x: calculate_news_weight(x), reverse=True)
            elif sort_by == "date":
                all_matches.sort(key=lambda x: x.get("date", ""), reverse=True)

            # Limit return count
            results = all_matches[:limit]

            # Build time range description (correctly determine if it's today)
            if start_date.date() == datetime.now().date() and start_date == end_date:
                time_range_desc = "today"
            elif start_date == end_date:
                time_range_desc = start_date.strftime("%Y-%m-%d")
            else:
                time_range_desc = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

            result = {
                "success": True,
                "summary": {
                    "total_found": len(all_matches),
                    "returned_count": len(results),
                    "requested_limit": limit,
                    "search_mode": search_mode,
                    "query": query,
                    "platforms": platforms or "all platforms",
                    "time_range": time_range_desc,
                    "sort_by": sort_by
                },
                "results": results
            }

            if search_mode == "fuzzy":
                result["summary"]["threshold"] = threshold
                if len(all_matches) < limit:
                    result["note"] = f"In fuzzy search mode, similarity threshold {threshold} only matched {len(all_matches)} results"

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

    def _search_by_keyword_mode(
        self,
        query: str,
        all_titles: Dict,
        id_to_name: Dict,
        current_date: datetime,
        include_url: bool
    ) -> List[Dict]:
        """
        Keyword search mode (exact matching)

        Args:
            query: Search keyword
            all_titles: All titles dictionary
            id_to_name: Platform ID to name mapping
            current_date: Current date

        Returns:
            Matched news list
        """
        matches = []
        query_lower = query.lower()

        for platform_id, titles in all_titles.items():
            platform_name = id_to_name.get(platform_id, platform_id)

            for title, info in titles.items():
                # Exact inclusion check
                if query_lower in title.lower():
                    news_item = {
                        "title": title,
                        "platform": platform_id,
                        "platform_name": platform_name,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "similarity_score": 1.0,  # Exact match, similarity is 1
                        "ranks": info.get("ranks", []),
                        "count": len(info.get("ranks", [])),
                        "rank": info["ranks"][0] if info["ranks"] else 999
                    }

                    # Conditionally add URL field
                    if include_url:
                        news_item["url"] = info.get("url", "")
                        news_item["mobileUrl"] = info.get("mobileUrl", "")

                    matches.append(news_item)

        return matches

    def _search_by_fuzzy_mode(
        self,
        query: str,
        all_titles: Dict,
        id_to_name: Dict,
        current_date: datetime,
        threshold: float,
        include_url: bool
    ) -> List[Dict]:
        """
        Fuzzy search mode (uses similarity algorithm)

        Args:
            query: Search content
            all_titles: All titles dictionary
            id_to_name: Platform ID to name mapping
            current_date: Current date
            threshold: Similarity threshold

        Returns:
            Matched news list
        """
        matches = []

        for platform_id, titles in all_titles.items():
            platform_name = id_to_name.get(platform_id, platform_id)

            for title, info in titles.items():
                # Fuzzy matching
                is_match, similarity = self._fuzzy_match(query, title, threshold)

                if is_match:
                    news_item = {
                        "title": title,
                        "platform": platform_id,
                        "platform_name": platform_name,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "similarity_score": round(similarity, 4),
                        "ranks": info.get("ranks", []),
                        "count": len(info.get("ranks", [])),
                        "rank": info["ranks"][0] if info["ranks"] else 999
                    }

                    # Conditionally add URL field
                    if include_url:
                        news_item["url"] = info.get("url", "")
                        news_item["mobileUrl"] = info.get("mobileUrl", "")

                    matches.append(news_item)

        return matches

    def _search_by_entity_mode(
        self,
        query: str,
        all_titles: Dict,
        id_to_name: Dict,
        current_date: datetime,
        include_url: bool
    ) -> List[Dict]:
        """
        Entity search mode (automatically sorted by weight)

        Args:
            query: Entity name
            all_titles: All titles dictionary
            id_to_name: Platform ID to name mapping
            current_date: Current date

        Returns:
            Matched news list
        """
        matches = []

        for platform_id, titles in all_titles.items():
            platform_name = id_to_name.get(platform_id, platform_id)

            for title, info in titles.items():
                # Entity search: exactly contains entity name
                if query in title:
                    news_item = {
                        "title": title,
                        "platform": platform_id,
                        "platform_name": platform_name,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "similarity_score": 1.0,
                        "ranks": info.get("ranks", []),
                        "count": len(info.get("ranks", [])),
                        "rank": info["ranks"][0] if info["ranks"] else 999
                    }

                    # Conditionally add URL field
                    if include_url:
                        news_item["url"] = info.get("url", "")
                        news_item["mobileUrl"] = info.get("mobileUrl", "")

                    matches.append(news_item)

        return matches

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts

        Args:
            text1: Text 1
            text2: Text 2

        Returns:
            Similarity score (between 0-1)
        """
        # Use difflib.SequenceMatcher to calculate sequence similarity
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _fuzzy_match(self, query: str, text: str, threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Fuzzy matching function

        Args:
            query: Query text
            text: Text to match
            threshold: Matching threshold

        Returns:
            (whether matched, similarity score)
        """
        # Direct inclusion check
        if query.lower() in text.lower():
            return True, 1.0

        # Calculate overall similarity
        similarity = self._calculate_similarity(query, text)
        if similarity >= threshold:
            return True, similarity

        # Partial matching after word segmentation
        query_words = set(self._extract_keywords(query))
        text_words = set(self._extract_keywords(text))

        if not query_words or not text_words:
            return False, 0.0

        # Calculate keyword overlap
        common_words = query_words & text_words
        keyword_overlap = len(common_words) / len(query_words)

        if keyword_overlap >= 0.5:  # 50% keyword overlap
            return True, keyword_overlap

        return False, similarity

    def _extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        Extract keywords from text

        Args:
            text: Input text
            min_length: Minimum word length

        Returns:
            Keyword list
        """
        # Remove URLs and special characters
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracket content

        # Use regex for word segmentation (English)
        words = re.findall(r'[\w]+', text)

        # Filter stopwords and short words
        keywords = [
            word for word in words
            if word and len(word) >= min_length and word not in self.stopwords
        ]

        return keywords

    def _calculate_keyword_overlap(self, keywords1: List[str], keywords2: List[str]) -> float:
        """
        Calculate overlap between two keyword lists

        Args:
            keywords1: Keyword list 1
            keywords2: Keyword list 2

        Returns:
            Overlap score (between 0-1)
        """
        if not keywords1 or not keywords2:
            return 0.0

        set1 = set(keywords1)
        set2 = set(keywords2)

        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def search_related_news_history(
        self,
        reference_text: str,
        time_preset: str = "yesterday",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        threshold: float = 0.4,
        limit: int = 50,
        include_url: bool = False
    ) -> Dict:
        """
        Search for news related to given news in historical data

        Args:
            reference_text: Reference news title or content
            time_preset: Time range preset, optional:
                - "yesterday": Yesterday
                - "last_week": Last week (7 days)
                - "last_month": Last month (30 days)
                - "custom": Custom date range (need to provide start_date and end_date)
            start_date: Custom start date (only effective when time_preset="custom")
            end_date: Custom end date (only effective when time_preset="custom")
            threshold: Similarity threshold (between 0-1), default 0.4
            limit: Return count limit, default 50
            include_url: Whether to include URL links, default False (to save tokens)

        Returns:
            Search result dictionary, containing related news list

        Example:
            >>> tools = SearchTools()
            >>> result = tools.search_related_news_history(
            ...     reference_text="AI technology breakthrough",
            ...     time_preset="last_week",
            ...     threshold=0.4,
            ...     limit=50
            ... )
            >>> for news in result['results']:
            ...     print(f"{news['date']}: {news['title']} (similarity: {news['similarity_score']})")
        """
        try:
            # Parameter validation
            reference_text = validate_keyword(reference_text)
            threshold = max(0.0, min(1.0, threshold))
            limit = validate_limit(limit, default=50)

            # Determine query date range
            today = datetime.now()

            if time_preset == "yesterday":
                search_start = today - timedelta(days=1)
                search_end = today - timedelta(days=1)
            elif time_preset == "last_week":
                search_start = today - timedelta(days=7)
                search_end = today - timedelta(days=1)
            elif time_preset == "last_month":
                search_start = today - timedelta(days=30)
                search_end = today - timedelta(days=1)
            elif time_preset == "custom":
                if not start_date or not end_date:
                    raise InvalidParameterError(
                        "Custom time range requires start_date and end_date",
                        suggestion="Please provide start_date and end_date parameters"
                    )
                search_start = start_date
                search_end = end_date
            else:
                raise InvalidParameterError(
                    f"Unsupported time range: {time_preset}",
                    suggestion="Please use 'yesterday', 'last_week', 'last_month' or 'custom'"
                )

            # Extract keywords from reference text
            reference_keywords = self._extract_keywords(reference_text)

            if not reference_keywords:
                raise InvalidParameterError(
                    "Unable to extract keywords from reference text",
                    suggestion="Please provide more detailed text content"
                )

            # Collect all related news
            all_related_news = []
            current_date = search_start

            while current_date <= search_end:
                try:
                    # Read data for this date
                    all_titles, id_to_name, _ = self.data_service.parser.read_all_titles_for_date(current_date)

                    # Search related news
                    for platform_id, titles in all_titles.items():
                        platform_name = id_to_name.get(platform_id, platform_id)

                        for title, info in titles.items():
                            # Calculate title similarity
                            title_similarity = self._calculate_similarity(reference_text, title)

                            # Extract title keywords
                            title_keywords = self._extract_keywords(title)

                            # Calculate keyword overlap
                            keyword_overlap = self._calculate_keyword_overlap(
                                reference_keywords,
                                title_keywords
                            )

                            # Combined similarity (70% keyword overlap + 30% text similarity)
                            combined_score = keyword_overlap * 0.7 + title_similarity * 0.3

                            if combined_score >= threshold:
                                news_item = {
                                    "title": title,
                                    "platform": platform_id,
                                    "platform_name": platform_name,
                                    "date": current_date.strftime("%Y-%m-%d"),
                                    "similarity_score": round(combined_score, 4),
                                    "keyword_overlap": round(keyword_overlap, 4),
                                    "text_similarity": round(title_similarity, 4),
                                    "common_keywords": list(set(reference_keywords) & set(title_keywords)),
                                    "rank": info["ranks"][0] if info["ranks"] else 0
                                }

                                # Conditionally add URL field
                                if include_url:
                                    news_item["url"] = info.get("url", "")
                                    news_item["mobileUrl"] = info.get("mobileUrl", "")

                                all_related_news.append(news_item)

                except DataNotFoundError:
                    # No data for this date, continue to next day
                    pass
                except Exception as e:
                    # Log error but continue processing other dates
                    print(f"Warning: Error processing date {current_date.strftime('%Y-%m-%d')}: {e}")

                # Move to next day
                current_date += timedelta(days=1)

            if not all_related_news:
                return {
                    "success": True,
                    "results": [],
                    "total": 0,
                    "query": reference_text,
                    "time_preset": time_preset,
                    "date_range": {
                        "start": search_start.strftime("%Y-%m-%d"),
                        "end": search_end.strftime("%Y-%m-%d")
                    },
                    "message": "No related news found"
                }

            # Sort by similarity
            all_related_news.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Limit return count
            results = all_related_news[:limit]

            # Statistics
            platform_distribution = Counter([news["platform"] for news in all_related_news])
            date_distribution = Counter([news["date"] for news in all_related_news])

            result = {
                "success": True,
                "summary": {
                    "total_found": len(all_related_news),
                    "returned_count": len(results),
                    "requested_limit": limit,
                    "threshold": threshold,
                    "reference_text": reference_text,
                    "reference_keywords": reference_keywords,
                    "time_preset": time_preset,
                    "date_range": {
                        "start": search_start.strftime("%Y-%m-%d"),
                        "end": search_end.strftime("%Y-%m-%d")
                    }
                },
                "results": results,
                "statistics": {
                    "platform_distribution": dict(platform_distribution),
                    "date_distribution": dict(date_distribution),
                    "avg_similarity": round(
                        sum([news["similarity_score"] for news in all_related_news]) / len(all_related_news),
                        4
                    ) if all_related_news else 0.0
                }
            }

            if len(all_related_news) < limit:
                result["note"] = f"With relevance threshold {threshold}, only found {len(all_related_news)} related news items"

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
