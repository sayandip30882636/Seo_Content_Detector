import pandas as pd
import regex as re

def parse_html_regex(html_content):
    """
    Parses HTML using regex to extract title and clean body text.
    Handles errors and missing content gracefully.
    """
    if pd.isna(html_content) or not html_content.strip():
        return "", "", 0

    try:
        # 1. Extract Title
        # re.DOTALL makes '.' match newlines
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        title_text = title_match.group(1).strip() if title_match else ""

        # 2. Extract Body Text
        body_text = ""

        # Try to find <article> or <main> content first
        article_match = re.search(r'<article[^>]*>(.*?)</article>', html_content, re.IGNORECASE | re.DOTALL)
        main_match = re.search(r'<main[^>]*>(.*?)</main>', html_content, re.IGNORECASE | re.DOTALL)

        content_to_parse = ""
        if article_match:
            content_to_parse = article_match.group(1)
        elif main_match:
            content_to_parse = main_match.group(1)
        else:
            # Fallback to the entire body
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.IGNORECASE | re.DOTALL)
            if body_match:
                content_to_parse = body_match.group(1)
            else:
                # Last resort: use all HTML
                content_to_parse = html_content

        # Extract text from <p> tags within the selected content
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', content_to_parse, re.IGNORECASE | re.DOTALL)
        if paragraphs:
            body_text = ' '.join(paragraphs)
        else:
            # If no <p> tags found, use the whole content_to_parse
            body_text = content_to_parse

        # Clean the extracted text:
        # 1. Remove script and style tags (including their content)
        body_text = re.sub(r'<(script|style).*?>.*?</>', ' ', body_text, re.IGNORECASE | re.DOTALL)
        # 2. Remove all other HTML tags
        body_text = re.sub(r'<[^>]+>', ' ', body_text)
        # 3. Replace common HTML entities
        body_text = re.sub(r'&nbsp;', ' ', body_text)
        body_text = re.sub(r'&amp;', '&', body_text)
        body_text = re.sub(r'&lt;', '<', body_text)
        body_text = re.sub(r'&gt;', '>', body_text)
        # 4. Remove extra whitespace
        body_text = re.sub(r'\s+', ' ', body_text).strip()

        # 3. Calculate Word Count
        word_count = len(body_text.split())

        return title_text, body_text, word_count

    except Exception as e:
        # In a real app, you might log this error
        return "", "", 0
