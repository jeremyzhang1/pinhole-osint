from fastapi import APIRouter, status
import shodan

router = APIRouter()

SHODAN_API_KEY = ""

shodan_api = shodan.Shodan(SHODAN_API_KEY)

@router.get("/", tags=['client'], status_code=status.HTTP_200_OK)
async def healthCheck():
    return {"message": "API OK"}

@router.get("/get-original-response", tags=['client'], status_code=status.HTTP_200_OK)
async def getOriginalResponse(request_ip: str):
    print(request_ip)
    ipinfo = shodan_api.host(request_ip)
    return ipinfo
