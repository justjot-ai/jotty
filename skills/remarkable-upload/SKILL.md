# reMarkable Upload Skill

## Description
Uploads PDFs to reMarkable cloud for syncing to device. Supports registration, upload, and status checking.


## Type
base


## Capabilities
- communicate

## Tools

### upload_to_remarkable_tool
Upload a PDF file to reMarkable cloud.

**Parameters:**
- `pdf_path` (str, required): Path to PDF file to upload
- `folder` (str, optional): reMarkable folder path, default: '/'
- `force` (bool, optional): Overwrite if exists, default: False

**Returns:**
- `success` (bool): Whether upload succeeded
- `document_id` (str): reMarkable document ID
- `path` (str): Path on reMarkable device
- `error` (str, optional): Error message if failed

### register_remarkable_tool
Register device with reMarkable cloud using one-time code.

**Parameters:**
- `one_time_code` (str, required): 8-character code from https://my.remarkable.com/device/browser/connect

**Returns:**
- `success` (bool): Whether registration succeeded
- `message` (str): Status message
- `error` (str, optional): Error message if failed

### check_remarkable_status_tool
Check reMarkable connection status.

**Parameters:**
- None

**Returns:**
- `success` (bool): Whether check succeeded
- `registered` (bool): Whether device is registered
- `connected` (bool): Whether currently connected
- `error` (str, optional): Error message if failed

## Triggers
- "remarkable upload"
- "send to remarkable"
- "remarkable tablet"
- "upload"

## Category
workflow-automation
