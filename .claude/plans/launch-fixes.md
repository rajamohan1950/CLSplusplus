# CLS++ Launch Readiness Plan

## 1. Perfect Context Injection Prompt (4 files)
Unify all injection formats to one canonical prompt:
```
The following facts were learned about this user from prior conversations. Integrate them naturally — reference only when relevant, never repeat verbatim unless asked:
- fact1
- fact2
```
Files: `prototype/server.py` (3 locations), `extension/content_common.js` (buildContext)

## 2. Add Gemini to Demo Chat
Add third branch in `/api/chat/{uid}` using `google.generativeai`. Load `CLS_GOOGLE_API_KEY` from env.

## 3. User-Defined Labels
- Backend: PATCH `/api/memories/{uid}/{id}/labels`, add `labels` to response, filter param
- Frontend: label pills on cards, "+ label" button, sidebar label section

## 4. Sort/Filter
- Backend: `sort`, `order`, `layer`, `min_strength` params on `/api/memories`
- Frontend: sort dropdown, clickable layer items

## 5. Performance
- Remove debug print in `/api/context`
- Move inline imports to top level
- Async disk save
