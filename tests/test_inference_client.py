import io
import unittest
from unittest.mock import patch

from tools.inference.client import _text, interactive, request


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return b'{"status":"ok"}'


class InferenceClientTest(unittest.TestCase):
    def test_request_uses_json_http_api(self) -> None:
        with patch("urllib.request.urlopen", return_value=_Response()) as urlopen:
            result = request(
                "http://compute:5000/",
                "/v1/completions",
                {"prompt": "hello", "max_tokens": 8},
            )

        self.assertEqual(result, {"status": "ok"})
        sent = urlopen.call_args.args[0]
        self.assertEqual(sent.full_url, "http://compute:5000/v1/completions")
        self.assertEqual(sent.method, "POST")
        self.assertIn(b'"prompt": "hello"', sent.data)

    def test_extracts_completion_and_chat_text(self) -> None:
        self.assertEqual(_text({"choices": [{"text": "done"}]}), "done")
        self.assertEqual(
            _text({"choices": [{"message": {"content": "answer"}}]}), "answer"
        )

    def test_interactive_sends_multiple_prompts_in_one_process(self) -> None:
        responses = [
            {"choices": [{"text": "one"}]},
            {"choices": [{"text": "two"}]},
        ]
        with (
            patch("builtins.input", side_effect=["first", "second", "/quit"]),
            patch("tools.inference.client.request", side_effect=responses) as send,
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            interactive(
                "http://compute:5000",
                chat=False,
                system=None,
                max_tokens=8,
                temperature=0,
            )

        self.assertEqual(send.call_count, 2)
        self.assertIn("one", output.getvalue())
        self.assertIn("two", output.getvalue())


if __name__ == "__main__":
    unittest.main()
