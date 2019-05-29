#Sketchfab supports the Oauth2 protocol for authentication and authorization
### Introduction
#- Please refer to the manual for missing/relevant information : https://tools.ietf.org/html/draft-ietf-oauth-v2-31
#- This code sample does not work such as it is, it is a template that roughly shows how the token exchange works.
#- This code sample is written in python but the same logic applies for every other language
### Requirements
#To begin, obtain Oauth2 credentials from support@sketchfab.com.
#You must provide us a redirect uri to which we can redirect your calls
### Implementation
#The protocol works as follow:
#1. You ask for an authorization code from the Sketchfab server with a supplied `redirect_uri`
#2. Sketchfab asks permission to your user
#3. A successful authorization will pass the client the authorization code in the URL via the supplied `redirect_uri`
#4. You exchange this authorization code with an access token from the Sketchfab server
#5. You use the access token to authenticate and authorize your user

from requests_oauthlib import OAuth2Session
import requests
import webbrowser
import urllib.request
import unittest
from selenium import webdriver
import mechanize
import re
import time
import csv
import json

#def test_authentication_popup(wurl):
 #       driver = webdriver.Chrome()
  #      driver.implicitly_wait(30)
        # open webpage
  #      driver.get(wurl)
        # verify the title
  #      if(driver.title == "Authentication Successful"):
  #          print("Test Passed")
  #      else:
  #          print("Test failed")

#executable_path=r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'

CLIENT_ID = '097dae06-f5f8-43c4-b556-4ebfff71858e'
CLIENT_SECRET = 'uHgBN/q4hCWPT{!5o3=nP+VW)y*=?^>lIxhTTc*herO'
AUTHORIZE_URL = "https://login.microsoftonline.com/2ed5574c-f9ba-4426-9658-e477ad7439db/oauth2/v2.0/authorize"
ACCESS_TOKEN_URL = "https://login.microsoftonline.com/2ed5574c-f9ba-4426-9658-e477ad7439db/oauth2/v2.0/token"
scope = ['User.ReadBasic.All Mail.Read Mail.ReadWrite']
REDIRECT_URI = 'http://localhost:50619/'     # Should match Site URL

#CLIENT_ID = "YOUR_CLIENT_ID"
#CLIENT_SECRET = "YOUR_CLIENT_SECRET"

#REDIRECT_URI = 'https://your-website.com/oauth2_redirect'
#AUTHORIZE_URL = "https://sketchfab.com/oauth2/authorize/"
#ACCESS_TOKEN_URL = "https://sketchfab.com/oauth2/token/"

# 1. Ask for an authorization code
code1 = requests.get('{}?response_type=code&client_id={}&redirect_uri={}'.format(AUTHORIZE_URL, CLIENT_ID, REDIRECT_URI))
outlook = OAuth2Session(CLIENT_ID,scope=scope,redirect_uri=REDIRECT_URI)

# Redirect  the user owner to the OAuth provider (i.e. Outlook) using an URL with a few key OAuth parameters.
authorization_url_1, state = outlook.authorization_url(AUTHORIZE_URL)
#print ('Please go here and authorize,', authorization_url_1)
webbrowser.open(authorization_url_1)
#br = mechanize.urlopen(authorization_url_1)
#br.read()
#br.open(authorization_url_1).select_form()
#time.sleep(5)
#print(webbrowser.get()
#print(br.read())
#webbrowser.open(authorization_url_1)
#test_authentication_popup(authorization_url_1)
#driver = webdriver.Chrome(executable_path=r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe')
#driver.implicitly_wait(30)
#driver.get(authorization_url_1)

#print(content)
print(" ")
#print(code1.text)
# 2. The user logs in, accepts your client authentication request

# 3. Sketchfab redirects to your provided `redirect_uri` with the authorization code in the URL
# Ex : https://website.com/oauth2_redirect?code=123456789

# 4. Grab that code and exchange it for an `access_token`
Acode = input("Agregue el codigo:" )
code = requests.post(
    ACCESS_TOKEN_URL,
    data={
        'grant_type': 'authorization_code',
        'code': Acode,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'scope': scope
    }
)

print(" ")
print(code.text)
print(" ")
print(type(code.text))
List_Code = code.text.split(',')
List_Token_Code_extracted = List_Code[4].split(':')
Token_Code_final = List_Token_Code_extracted[1].replace('"','').replace('}','')
print(Token_Code_final)
print(" ")

Bearer = "Bearer " + Token_Code_final
print(Bearer)

# The response body of this request contains the `access_token` necessary to authenticate your requests
# Ex : {"access_token": "1234", "token_type": "Bearer", "expires_in": 36000, "refresh_token": "5678", "scope": "read write"}
# - expires_in => seconds to live for this `access_token`
# - refresh_token => A token used to fetch a new `access_token` (See below)


# Now you're all set, the following request shows how to use your `access_token` in your requests
# If your access token is recognized, this will return information regarding the current user
#requests.get('https://sketchfab.com/v2/users/me', headers={'Authorization': 'Bearer YOUR_ACCESS_TOKEN'})

# Extra:

# Before your access token expires, you can refresh it with the `refresh_token`. If it has expired,
# you will have to re-do the auhorization workflow
#requests.post(
#    ACCESS_TOKEN_URL,
#    data={
#        'grant_type': 'refresh_token',
#        'client_id': CLIENT_ID,
#        'client_secret': CLIENT_SECRET,
#        'refresh_token': 'YOUR_REFRESH_TOKEN'
#    }
#)
# The response body of this request is exactly the same as the one to get an access_token
# Ex : {"access_token": "1234", "token_type": "Bearer", "expires_in": 36000, "refresh_token": "5678", "scope": "read write"}
url = "https://graph.microsoft.com/beta/me/messages?$Top=700"

payload = ""
headers = {
    'Authorization': Bearer,
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Host': "graph.microsoft.com",
    'accept-encoding': "gzip, deflate",
    'Connection': "keep-alive",
    'cache-control': "no-cache",
    'contentType' : "row"
    }
#'Postman-Token': "291244b7-7754-45ba-bc49-2d2fa858546f,64d3523e-1735-4b9a-8af2-f3af3e3ad328",
# 'User-Agent': "PostmanRuntime/7.11.0",
response = requests.request("GET", url, data=payload, headers=headers)
#print(response.text)
print(" ")
print(type(response))
print(" ")
print(type(response.json))
print(" ")
#print(response.json())
print(type(response.json()))
Correos_PS = response.text.split('}}')
#Archivo1 = open("Correos2.csv","w")
#with Archivo1:
#    writer = csv.writer(Archivo1)
 #   writer.write(Correos_PS)
aJson = response.json()
#print(type(aJson))
print(aJson.keys())
#aJson.pop('@odata.context')
print(aJson.keys())
Archivo = open("CorreosPrueba1.json","w")
json = json.dumps(response.json().get('value'))
Archivo.write(json)
Archivo.close()
