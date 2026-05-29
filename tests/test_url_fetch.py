import unittest

from imperia_matching_tfm.property_extraction.url_fetch import detect_block_provider, html_to_text, is_blocked_response


class UrlFetchTest(unittest.TestCase):
    def test_detects_captcha_block(self) -> None:
        html = "<p>Please enable JS and disable any ad blocker</p><script src='captcha-delivery'></script>"

        self.assertTrue(is_blocked_response(html))
        self.assertEqual(detect_block_provider(html), "DataDome")

    def test_html_to_text_removes_scripts(self) -> None:
        html = "<html><body><h1>Piso en venta</h1><script>alert('x')</script><p>290.000€</p></body></html>"

        text = html_to_text(html)

        self.assertIn("Piso en venta", text)
        self.assertIn("290.000€", text)
        self.assertNotIn("alert", text)


if __name__ == "__main__":
    unittest.main()
