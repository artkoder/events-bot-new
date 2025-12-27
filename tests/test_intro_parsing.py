"""
Tests for intro parsing logic used in video generation.
Verifies that parse_intro_to_visual_lines correctly handles various text formats.
"""
import pytest


# Copy of the parsing code for testing (isolated from moviepy dependencies)
PREPOSITIONS = {'в', 'на', 'с', 'к', 'о', 'у', 'за', 'по', 'из', 'от', 'для', 'до', 'без', 'при', 'об', 'над', 'под'}


def parse_intro_to_visual_lines(intro_data):
    """
    Parses intro_data into visual line structure for video rendering.
    Supports: text, date, cities, count (embedded number).
    """
    result = []
    
    text = intro_data.get('text', '').upper()
    count = str(intro_data.get('count', ''))
    date_str = intro_data.get('date', '')
    cities = intro_data.get('cities', [])
    
    if not text:
        text = 'СОБЫТИЯ'
    
    # Merge prepositions with next word
    words = text.split()
    merged_words = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.lower() in PREPOSITIONS and i + 1 < len(words):
            merged_words.append(w + ' ' + words[i + 1])
            i += 2
        else:
            merged_words.append(w)
            i += 1
    
    # Special case: "WORD NUMBER WORD" pattern
    has_number_pattern = False
    if count:
        for idx, word in enumerate(merged_words):
            if word == count or word.isdigit():
                has_number_pattern = True
                break
    
    if has_number_pattern:
        number_idx = None
        for idx, word in enumerate(merged_words):
            if word == count or word.isdigit():
                number_idx = idx
                break
        
        if number_idx is not None:
            line_with_number = []
            if number_idx > 0:
                line_with_number.append({'text': merged_words[number_idx - 1], 'type': 'text', 'size': 'medium'})
            line_with_number.append({'text': count, 'type': 'number', 'size': 'large'})
            if number_idx + 1 < len(merged_words):
                line_with_number.append({'text': merged_words[number_idx + 1], 'type': 'text', 'size': 'medium'})
            result.append(line_with_number)
            
            remaining_before = merged_words[:max(0, number_idx - 1)]
            remaining_after = merged_words[number_idx + 2:]
            remaining = remaining_before + remaining_after
            
            current_line = []
            for word in remaining:
                is_long = len(word) >= 7
                if is_long:
                    if current_line:
                        result.append(current_line)
                        current_line = []
                    result.append([{'text': word, 'type': 'text', 'size': 'large'}])
                else:
                    current_line.append({'text': word, 'type': 'text', 'size': 'medium'})
                    if len(current_line) >= 3:
                        result.append(current_line)
                        current_line = []
            if current_line:
                result.append(current_line)
    else:
        current_line = []
        for word in merged_words:
            is_long = len(word) >= 7
            if is_long:
                if current_line:
                    result.append(current_line)
                    current_line = []
                result.append([{'text': word, 'type': 'text', 'size': 'large'}])
            else:
                current_line.append({'text': word, 'type': 'text', 'size': 'medium'})
                if len(current_line) >= 3:
                    result.append(current_line)
                    current_line = []
        if current_line:
            result.append(current_line)
    
    if date_str:
        result.append([{'text': date_str.upper(), 'type': 'date', 'size': 'large'}])
    
    if cities:
        cities_text = ', '.join(cities)
        result.append([{'text': cities_text, 'type': 'cities', 'size': 'small'}])
    
    return result


class TestIntroParsingBasic:
    """Basic parsing functionality tests."""
    
    def test_empty_text_defaults_to_events(self):
        result = parse_intro_to_visual_lines({'text': '', 'date': '', 'cities': [], 'count': ''})
        
        assert len(result) == 1
        assert result[0][0]['text'] == 'СОБЫТИЯ'
    
    def test_simple_text_parsed(self):
        result = parse_intro_to_visual_lines({'text': 'СЕМЕЙНЫЕ ВЫХОДНЫЕ', 'date': '', 'cities': [], 'count': ''})
        
        # Both words are 7+ chars so each goes on separate line
        assert len(result) == 2
        assert result[0][0]['text'] == 'СЕМЕЙНЫЕ'
        assert result[1][0]['text'] == 'ВЫХОДНЫЕ'
    
    def test_short_words_grouped_on_line(self):
        result = parse_intro_to_visual_lines({'text': 'КУДА ПОЙТИ', 'date': '', 'cities': [], 'count': ''})
        
        # Short words should be on same line
        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0]['text'] == 'КУДА'
        assert result[0][1]['text'] == 'ПОЙТИ'


class TestIntroPrepositions:
    """Preposition merging tests."""
    
    def test_preposition_merged_with_next_word(self):
        result = parse_intro_to_visual_lines({'text': 'КУДА ПОЙТИ С ДРУЗЬЯМИ', 'date': '', 'cities': [], 'count': ''})
        
        # "С ДРУЗЬЯМИ" should be merged
        all_texts = [e['text'] for line in result for e in line]
        assert 'С ДРУЗЬЯМИ' in all_texts
        assert 'С' not in all_texts  # Should not be separate
    
    def test_multiple_prepositions_merged(self):
        result = parse_intro_to_visual_lines({'text': 'ИДЕИ ДЛЯ КУЛЬТУРНОГО ОТДЫХА', 'date': '', 'cities': [], 'count': ''})
        
        all_texts = [e['text'] for line in result for e in line]
        assert 'ДЛЯ КУЛЬТУРНОГО' in all_texts
    
    def test_preposition_at_end_not_merged(self):
        # Preposition at end has nothing to merge with
        result = parse_intro_to_visual_lines({'text': 'КУДА НА', 'date': '', 'cities': [], 'count': ''})
        
        all_texts = [e['text'] for line in result for e in line]
        assert 'НА' in all_texts  # Should remain separate


class TestIntroNumberPattern:
    """Number pattern detection and rendering tests."""
    
    def test_number_pattern_detected(self):
        result = parse_intro_to_visual_lines({
            'text': 'ПОДОБРАЛИ 5 СОБЫТИЙ НА ВЫХОДНЫЕ',
            'date': '',
            'cities': [],
            'count': '5'
        })
        
        # First line should contain ПОДОБРАЛИ + 5 + СОБЫТИЙ
        first_line = result[0]
        types = [e['type'] for e in first_line]
        texts = [e['text'] for e in first_line]
        
        assert 'number' in types
        assert '5' in texts
        assert len(first_line) == 3  # word before, number, word after
    
    def test_number_with_correct_types(self):
        result = parse_intro_to_visual_lines({
            'text': 'ПОДОБРАЛИ 5 СОБЫТИЙ',
            'date': '',
            'cities': [],
            'count': '5'
        })
        
        first_line = result[0]
        assert first_line[0]['type'] == 'text'  # ПОДОБРАЛИ
        assert first_line[1]['type'] == 'number'  # 5
        assert first_line[2]['type'] == 'text'  # СОБЫТИЙ


class TestIntroDatesAndCities:
    """Date and cities rendering tests."""
    
    def test_date_added_as_separate_line(self):
        result = parse_intro_to_visual_lines({
            'text': 'СЕМЕЙНЫЕ ВЫХОДНЫЕ',
            'date': '27-28 ДЕКАБРЯ',
            'cities': [],
            'count': ''
        })
        
        # Date should be last line
        last_line = result[-1]
        assert last_line[0]['type'] == 'date'
        assert last_line[0]['text'] == '27-28 ДЕКАБРЯ'
    
    def test_cities_added_after_date(self):
        result = parse_intro_to_visual_lines({
            'text': 'СЕМЕЙНЫЕ ВЫХОДНЫЕ',
            'date': '27 ДЕКАБРЯ',
            'cities': ['Калининград', 'Светлогорск'],
            'count': ''
        })
        
        # Cities should be very last
        last_line = result[-1]
        assert last_line[0]['type'] == 'cities'
        assert 'Калининград' in last_line[0]['text']
        assert 'Светлогорск' in last_line[0]['text']
    
    def test_multiple_cities_joined_with_comma(self):
        result = parse_intro_to_visual_lines({
            'text': 'СОБЫТИЯ',
            'date': '',
            'cities': ['Калининград', 'Светлогорск', 'Зеленоградск'],
            'count': ''
        })
        
        cities_line = result[-1]
        assert cities_line[0]['text'] == 'Калининград, Светлогорск, Зеленоградск'


class TestIntroEdgeCases:
    """Edge case handling tests."""
    
    def test_only_date(self):
        result = parse_intro_to_visual_lines({
            'text': '',
            'date': '26 ДЕКАБРЯ',
            'cities': [],
            'count': ''
        })
        
        assert len(result) == 2  # Default text + date
        assert result[0][0]['text'] == 'СОБЫТИЯ'
        assert result[1][0]['type'] == 'date'
    
    def test_only_cities(self):
        result = parse_intro_to_visual_lines({
            'text': '',
            'date': '',
            'cities': ['Москва'],
            'count': ''
        })
        
        assert len(result) == 2  # Default text + cities
        assert result[-1][0]['type'] == 'cities'
    
    def test_lowercase_preserved_as_uppercase(self):
        result = parse_intro_to_visual_lines({
            'text': 'куда пойти',
            'date': '',
            'cities': [],
            'count': ''
        })
        
        all_texts = [e['text'] for line in result for e in line]
        assert 'КУДА' in all_texts
        assert 'ПОЙТИ' in all_texts
