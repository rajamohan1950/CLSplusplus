# CLS++ AWS Free Tier — Step-by-Step (No Experience Needed)

**Cost: $0 for 12 months** (new AWS accounts)

---

## What You Need

1. An AWS account (create one at [aws.amazon.com](https://aws.amazon.com) — free)
2. This file: `cloudformation-free-tier.yaml`

---

## Steps (5 minutes)

### 1. Log in to AWS

1. Go to [console.aws.amazon.com](https://console.aws.amazon.com)
2. Sign in with your email and password

### 2. Open CloudFormation

1. In the search bar at the top, type **CloudFormation**
2. Click **CloudFormation** (under Services)

### 3. Create a New Stack

1. Click the orange **Create stack** button
2. Choose **With new resources (standard)**

### 4. Upload the Template

1. Under "Prepare template", select **Upload a template file**
2. Click **Choose file**
3. Select `cloudformation-free-tier.yaml` from your computer
4. Click **Next**

### 5. Enter Your Password

1. **Stack name:** Leave as is or type `clsplusplus` (any name is fine)
2. **DbPassword:** Type a password you'll remember  
   - Example: `MySecurePass123!`  
   - Must be at least 8 characters
3. Click **Next**

### 6. Click Through the Next Screens

1. Click **Next** (leave everything as default)
2. Scroll down and check the box: **I acknowledge that AWS CloudFormation might create IAM resources**
3. Click **Submit**

### 7. Wait (10–15 minutes)

1. You'll see "CREATE_IN_PROGRESS"
2. Go get a coffee — the first run takes 10–15 minutes (downloads and builds everything)
3. When it says **CREATE_COMPLETE**, you're done

### 8. Get Your API URL

1. Click your stack name
2. Click the **Outputs** tab
3. Copy the value next to **ApiUrl** (looks like `http://12.34.56.78:8080`)
4. Paste it in your browser — you should see the API welcome page

---

## Test That It Works

1. Open **HealthUrl** from the Outputs tab in your browser  
   - You should see something like: `{"status":"healthy","stores":{...}}`
2. Open **DocsUrl** to see the API documentation

---

## If Something Goes Wrong

- **Stack fails:** Check the Events tab for the error. Common: password too short (use 8+ chars).
- **Health check times out:** Wait 5 more minutes and try the HealthUrl again. First run is slow.
- **Can't connect:** Make sure the stack status is CREATE_COMPLETE before trying the URLs.

---

## Free Tier Limits (12 months)

- 750 hours/month of EC2 (t3.micro) — this uses ~720 hours = free
- 750 hours/month of RDS (db.t3.micro) — free
- 20 GB storage — free

**After 12 months:** You'll start getting a small bill (~$15/month) unless you delete the stack.

---

## To Delete Everything (Stop All Charges)

1. Go to CloudFormation
2. Select your stack
3. Click **Delete**
4. Confirm
