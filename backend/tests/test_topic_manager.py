import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock setup.init BEFORE importing TopicManager to avoid DB connection
mock_setup = MagicMock()
sys.modules["setup"] = MagicMock()
sys.modules["setup.init"] = mock_setup
mock_setup.EMBEDDINGS = MagicMock()

from utils.topic_manager import TopicManager


class TestTopicManager(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        mock_setup.reset_mock()

    def test_calculate_topic_similarity_high(self):
        # Setup mock to return identical embeddings
        mock_setup.EMBEDDINGS.embed_query.side_effect = lambda x: [1.0, 0.0, 0.0]

        result = TopicManager.calculate_topic_similarity(
            "What is Docker?", "Docker containers"
        )

        self.assertAlmostEqual(result["similarity_score"], 1.0)
        self.assertTrue(result["is_continuation"])
        self.assertEqual(result["confidence_level"], "high")

    def test_calculate_topic_similarity_low(self):
        # Setup mock to return orthogonal embeddings (0 similarity)
        def side_effect(text):
            if "Docker" in text:
                return [1.0, 0.0]
            else:
                return [0.0, 1.0]

        mock_setup.EMBEDDINGS.embed_query.side_effect = side_effect

        result = TopicManager.calculate_topic_similarity(
            "What is Docker?", "Banana recipe"
        )

        self.assertLess(result["similarity_score"], 0.1)
        self.assertFalse(result["is_continuation"])
        self.assertEqual(result["confidence_level"], "low")

    @patch("utils.topic_manager.get_chat_history")
    def test_get_relevant_context(self, mock_get_history):
        # Mock history
        mock_history = MagicMock()
        mock_msg1 = MagicMock()
        mock_msg1.content = "I like Docker"
        mock_msg1.type = "user"

        mock_msg2 = MagicMock()
        mock_msg2.content = "It is useful"
        mock_msg2.type = "assistant"

        # history with system(ignored) + msg1 + msg2
        mock_history.messages = [MagicMock(), mock_msg1, mock_msg2]
        mock_get_history.return_value = mock_history

        # Mock embeddings to make everything similar
        mock_setup.EMBEDDINGS.embed_query.return_value = [1.0, 0.0]

        context = TopicManager.get_relevant_context_for_continuation(
            "sess_1", "Docker?"
        )

        self.assertEqual(len(context), 2)
        self.assertEqual(context[0]["content"], "I like Docker")

    @patch("utils.topic_manager.get_graph_instance")
    def test_get_session_topic(self, mock_get_graph):
        mock_graph = MagicMock()
        mock_get_graph.return_value = mock_graph

        mock_graph.query.return_value = [{"topic": "Docker Basics"}]

        topic = TopicManager.get_session_topic("sess_1")
        self.assertEqual(topic, "Docker Basics")

    @patch("utils.topic_manager.get_graph_instance")
    def test_get_session_topic_empty(self, mock_get_graph):
        mock_graph = MagicMock()
        mock_get_graph.return_value = mock_graph

        mock_graph.query.return_value = []

        topic = TopicManager.get_session_topic("sess_1")
        self.assertEqual(topic, "")


if __name__ == "__main__":
    unittest.main()
