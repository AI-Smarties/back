from django.test import TestCase, Client

from django.urls import reverse

# Create your tests here.

class TestViews(TestCase):
    def setUp(self):
        self.message_url = reverse("message")
        self.client = Client()

    def test_message_GET(self):
        pass

    def test_message_POST(self):
        pass
