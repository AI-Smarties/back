import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def message_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        body = json.loads(request.body)
        text = body.get("text", "")
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({
        "reply": f"Sait backendilta vastauksen: {text}"
    })
