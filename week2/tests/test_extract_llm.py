# tests/test_extract_llm.py
import pytest
from ..app.services.extract import extract_action_items, extract_action_items_llm

def test_empty_string():
    """Test behavior with empty string input"""
    result = extract_action_items_llm("")
    assert isinstance(result, list)
    assert len(result) == 0
    assert result == extract_action_items("")

def test_narrative_text():
    """Test behavior with purely narrative text (no clear action items)"""
    text = "This is just a regular sentence. It describes something that happened yesterday."
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)
    assert result == original

def test_bullet_points():
    """Test behavior with bullet point lists"""
    text = """
    - Set up database
    * Implement API endpoint
    1. Write unit tests
    • Review code
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_keyword_prefixes():
    """Test behavior with keyword prefixes"""
    text = """
    TODO: Complete documentation
    Action: Schedule meeting with team
    Next: Prepare presentation slides
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    print(result)
    print(original)
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_mixed_content():
    """Test behavior with mixed content containing multiple action items"""
    text = """
    Meeting notes from 2023-10-15:
    
    - [ ] Set up database
    * implement API extract endpoint
    1. Write tests
    
    This is just a narrative sentence about what we discussed.
    
    Next: Review the design document
    Action: Send email to stakeholders
    
    Some more random text.
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    print(result)
    print(original)
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_complex_notes():
    """Test behavior with complex notes containing special characters, newlines, and indentation"""
    text = """
        Project Update:
        
        - [ ] Fix login issue (user can't authenticate)
        *   Update documentation for new feature
        3.  Investigate performance problem:
            - Check database queries
            - Profile API endpoints
            
        Action items from discussion:
        TODO: Contact client about timeline
        Next: Prepare demo for Friday
        
        Note: This is not an action item.
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    print(result)
    print(original) 
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_consistency_with_original():
    """Test that LLM version produces structurally compatible results with original function"""
    note = "TODO: 审核 PR\n- 准备下周会议\n* Fix bug in login flow\nNext: Send email to team"
    
    old_result = extract_action_items(note)
    new_result = extract_action_items_llm(note)
    
    # Verify both return the same type and structure
    assert isinstance(old_result, list)
    assert isinstance(new_result, list)
    assert all(isinstance(item, str) for item in old_result)
    assert all(isinstance(item, str) for item in new_result)
    
    # Verify similar result count and content
    assert len(old_result) == len(new_result)
    print(old_result)
    print(new_result) 
    # Verify semantic equivalence (key content matches)
    for orig_item in old_result:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in new_result)