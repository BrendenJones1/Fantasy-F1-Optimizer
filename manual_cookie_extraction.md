# Manual Fantasy F1 Cookie Extraction

## Step 1: Get Reese84 Cookie

```bash
curl --location 'https://api.formula1.com/6657193977244c13?d=account.formula1.com' \
--data '{"solution":{"interrogation":{"st":162229509,"sr":1959639815,"cr":78830557},"version":"stable"},"error":null,"performance":{"interrogation":185}}'
```

**Expected Response:**
```json
{
  "token": "3:MV4qXrMiVKIR3PxTstSArg==:VgtgpXyqA5j7e7RxtHjY+NhmgtEFhK1M7CkmupyooN8hNoSgTRubkwretretrI+0U4rfgfFIff+99mj+BRBb",
  "renewInSec": 743,
  "cookieDomain": ".formula1.com"
}
```

## Step 2: Login with Email/Password

```bash
curl --location 'https://api.formula1.com/v2/account/subscriber/authenticate/by-password' \
--header 'apiKey: fCUCjWxKPu9y1JwRAv8BpGLEgiAuThx7' \
--header 'Content-Type: application/json' \
--data '{"login":"your_email@example.com","password":"your_password"}'
```

**Expected Response:**
```json
{
  "subscriptionToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "12345",
    "email": "your_email@example.com"
  }
}
```

## Step 3: Use the Tokens

- **Reese84 Token**: Use for anti-bot detection
- **Subscription Token**: Use to generate X-F1-Cookie-Data
- **X-F1-Cookie-Data**: Use in API requests for authentication

## Important Notes

1. **Token Expiration**: Tokens expire (usually 1-2 hours)
2. **Rate Limiting**: Don't make too many requests
3. **Security**: Keep tokens secure, don't share them
4. **Refresh**: You'll need to refresh tokens periodically
