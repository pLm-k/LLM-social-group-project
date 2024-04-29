from vllm_seqeval2 import extract_annotation, postprocessing, convert_to_bio_format
import unittest
import numpy as np

#run command:  python -m unittest -v unit_test_vllm_seqeval.py
# or single: python unit_test_vllm_seqeval.py test_vllm_seqeval.test_extract_annotation_with_tags
#Fixes: change assert ==* in line 22, 26

class test_vllm_seqeval(unittest.TestCase):

    ## Test extract_annotation function
    def test_extract_annotation_with_tags(self):
        # Test with correct start and end tags
        input_text = "Some text <annotation>annotated content</annotation> more text"
        generated_text = "<annotation>annotated content</annotation>"
        assert extract_annotation(input_text, generated_text) == "annotated content"


    def test_extract_annotation_missing_tags(self):
        # Test with missing tags
        input_text = "Some text without proper tags"
        generated_text = "Some text without proper tags"
        assert extract_annotation(input_text, generated_text) == "++++empty+++++ " + generated_text

    def test_extract_annotation_empty_string(self):
        # Test with an empty string
        assert extract_annotation("<annotation></annotation>", "") ==  "++++empty+++++ "


    ## Test postprocessing function
    def test_postprocessing_varied_strings(self):
        # Test with different strings
        input_array = np.array(["This is a test.", "Another test, with punctuation!"])
        expected_output = np.array(["This is a test", "Another test with punctuation"])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)

    def test_postprocessing_special_characters(self):
        # Test with special characters
        input_array = np.array(["Special & characters? $%", "New-line\ncharacters"])
        expected_output = np.array(["Special  characters", "Newlinecharacters"])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)

    def test_postprocessing_mixed_case(self):
        # Test with mixed cases
        input_array = np.array(["MiXed CaSe tExt", "ANOTHER Mixed CASE"])
        expected_output = np.array(["MiXed CaSe tExt", "ANOTHER Mixed CASE"])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)

    def test_postprocessing_numbers_punctuation(self):
        # Test with numbers and punctuation
        input_array = np.array(["Text with numbers 12345", "Punctuation! Is it removed?"])
        expected_output = np.array(["Text with numbers 12345", "Punctuation Is it removed"])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)


    def test_postprocessing_unicode_characters(self):
        # Test with unicode characters
        input_array = np.array(["Unicode: 😀😃😄😁", "Test with é, ñ, and ç"])
        expected_output = np.array(["Unicode ", "Test with é, ñ, and ç"])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)


    def test_postprocessing_empty_strings(self):
        # Test with empty strings
        input_array = np.array(["", ""])
        expected_output = np.array(["", ""])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)

    def test_postprocessing_long_text(self):
        # Test with long text
        input_array = np.array(["This is a longer piece of text to see how the function handles it.", 
                            "Another long sentence, with, multiple, punctuation marks!"])
        expected_output = np.array(["This is a longer piece of text to see how the function handles it", 
                                "Another long sentence with multiple punctuation marks"])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)

    def test_postprocessing_empty_array(self):
        # Test with an empty array
        input_array = np.array([])
        expected_output = np.array([])
        np.testing.assert_array_equal(postprocessing(input_array), expected_output)


    ## Test convert_to_bio function
    def test_convert_to_bio_with_tokens(self):
        # Test with sequences having specific tokens
        seq = ["@@entity1##", "is", "an", "example", "@@entity2##"]
        expected_bio = ["B", "O", "O", "O", "B"]
        assert convert_to_bio_format(seq) == expected_bio

    def test_convert_to_bio_without_tokens(self):
        # Test with sequences without the tokens
        seq = ["This", "is", "a", "test"]
        expected_bio = ["O", "O", "O", "O"]
        assert convert_to_bio_format(seq) == expected_bio

    def test_convert_to_bio_empty_sequence(self):
        # Test with an empty sequence
        assert convert_to_bio_format([]) == []

    def test_convert_to_bio_nested_entities(self):
        # Test with nested entities
        seq = ["@@Outer", "@@Inner##", "entity##"]
        expected_bio = ["B", "B", "I"]
        assert convert_to_bio_format(seq) == expected_bio

    def test_convert_to_bio_continuous_entities(self):
        # Test with continuous entities
        seq = ["@@First##", "@@Second##", "text"]
        expected_bio = ["B", "B", "O"]
        assert convert_to_bio_format(seq) == expected_bio

    def test_convert_to_bio_multiple_I_annotations(self):
        # Test with multiple I annotations
        seq = ["This", "@@is", "a", "test##"]
        expected_bio = ["O", "B", "I", "I"]
        assert convert_to_bio_format(seq) == expected_bio


if __name__ == "__main__":
    unittest.main()









