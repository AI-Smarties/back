import json
from django.test import TestCase, Client
from django.urls import reverse


# Create your tests here.

class TestViews(TestCase):
    def setUp(self):
        self.message_url = reverse("message")
        self.client = Client()

    def test_message_get(self):
        response = self.client.get(self.message_url)

        self.assertEqual(response.status_code, 405)

    def test_message_post(self):
        data = {"text": "Test"}
        response = self.client.post(
            self.message_url, json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)

    def test_message_post_not_json(self):
        data = {"text": "Test"}
        response = self.client.post(
            self.message_url, data
        )

        self.assertEqual(response.status_code, 400)
