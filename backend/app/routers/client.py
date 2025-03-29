from fastapi import APIRouter, status
import requests

router = APIRouter()

@router.get("/", tags=['client'], status_code=status.HTTP_200_OK)
async def healthCheck():
    return {"message": "API OK"}

@router.get("/get-original-response", tags=['client'], status_code=status.HTTP_200_OK)
async def getOriginalResponse(request: str):
    pass

    # TODO: call shodan and get IP information