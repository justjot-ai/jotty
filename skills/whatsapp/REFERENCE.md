# WhatsApp Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`send_whatsapp_message_tool`](#send_whatsapp_message_tool) | Send a text message via WhatsApp. |
| [`send_whatsapp_image_tool`](#send_whatsapp_image_tool) | Send an image via WhatsApp. |
| [`send_whatsapp_document_tool`](#send_whatsapp_document_tool) | Send a document via WhatsApp. |
| [`send_whatsapp_template_tool`](#send_whatsapp_template_tool) | Send a template message via WhatsApp. |
| [`send_whatsapp_location_tool`](#send_whatsapp_location_tool) | Send a location via WhatsApp. |
| [`mark_message_read_tool`](#mark_message_read_tool) | Mark a message as read. |
| [`send_whatsapp_media_tool`](#send_whatsapp_media_tool) | Send media via WhatsApp with automatic provider selection. |
| [`get_whatsapp_status_tool`](#get_whatsapp_status_tool) | Get WhatsApp connection status. |
| [`send_whatsapp_video_tool`](#send_whatsapp_video_tool) | Send a video via WhatsApp. |
| [`send_whatsapp_audio_tool`](#send_whatsapp_audio_tool) | Send an audio message via WhatsApp. |
| [`send_whatsapp_reaction_tool`](#send_whatsapp_reaction_tool) | React to a WhatsApp message with an emoji. |
| [`send_whatsapp_reply_tool`](#send_whatsapp_reply_tool) | Reply to a specific WhatsApp message. |
| [`get_whatsapp_profile_tool`](#get_whatsapp_profile_tool) | Get WhatsApp Business profile information. |
| [`send_whatsapp_contacts_tool`](#send_whatsapp_contacts_tool) | Send contact cards via WhatsApp. |
| [`send_whatsapp_interactive_tool`](#send_whatsapp_interactive_tool) | Send interactive messages (buttons or lists) via WhatsApp. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`send_whatsapp_message`](#send_whatsapp_message) | Send WhatsApp message synchronously. |
| [`send_whatsapp_media`](#send_whatsapp_media) | Send WhatsApp media synchronously. |
| [`get_whatsapp_status`](#get_whatsapp_status) | Get WhatsApp status synchronously. |

---

## `send_whatsapp_message_tool`

Send a text message via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number with country code
- **message** (`str, required`): Message text
- **preview_url** (`bool, optional`): Enable URL preview (default: False)
- **phone_id** (`str, optional`): WhatsApp Phone Number ID
- **token** (`str, optional`): Access token

**Returns:** Dictionary with success, message_id, to

---

## `send_whatsapp_image_tool`

Send an image via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **image_url** (`str, optional`): URL of image to send
- **image_path** (`str, optional`): Local path to image
- **caption** (`str, optional`): Image caption

**Returns:** Dictionary with success, message_id, to

---

## `send_whatsapp_document_tool`

Send a document via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **document_url** (`str, optional`): URL of document
- **document_path** (`str, optional`): Local path to document
- **filename** (`str, optional`): Display filename
- **caption** (`str, optional`): Document caption

**Returns:** Dictionary with success, message_id, to

---

## `send_whatsapp_template_tool`

Send a template message via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **template_name** (`str, required`): Template name
- **language_code** (`str, optional`): Language code (default: 'en_US')
- **components** (`list, optional`): Template components

**Returns:** Dictionary with success, message_id, to, template

---

## `send_whatsapp_location_tool`

Send a location via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **latitude** (`float, required`): Latitude
- **longitude** (`float, required`): Longitude
- **name** (`str, optional`): Location name
- **address** (`str, optional`): Location address

**Returns:** Dictionary with success, message_id, to

---

## `mark_message_read_tool`

Mark a message as read.

**Parameters:**

- **message_id** (`str, required`): WhatsApp message ID to mark as read

**Returns:** Dictionary with success

---

## `send_whatsapp_media_tool`

Send media via WhatsApp with automatic provider selection.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **media_path** (`str, required`): Path to media file or URL
- **media_type** (`str, optional`): "image", "video", "audio", "document"
- **caption** (`str, optional`): Media caption
- **provider** (`str, optional`): "baileys", "business", or "auto"

**Returns:** Dictionary with success, message_id, provider

---

## `get_whatsapp_status_tool`

Get WhatsApp connection status.

**Parameters:**

- **provider** (`str, optional`): "baileys", "business", or "auto"

**Returns:** Dictionary with success, connected, phone, name, provider

---

## `send_whatsapp_video_tool`

Send a video via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **video_url** (`str, optional`): URL of video
- **video_path** (`str, optional`): Local path to video
- **caption** (`str, optional`): Video caption

**Returns:** Dictionary with success, message_id, to

---

## `send_whatsapp_audio_tool`

Send an audio message via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **audio_url** (`str, optional`): URL of audio
- **audio_path** (`str, optional`): Local path to audio

**Returns:** Dictionary with success, message_id, to

---

## `send_whatsapp_reaction_tool`

React to a WhatsApp message with an emoji.

**Parameters:**

- **message_id** (`str, required`): ID of message to react to
- **emoji** (`str, required`): Emoji to react with (e.g., thumbs up)

**Returns:** Dictionary with success, message_id

---

## `send_whatsapp_reply_tool`

Reply to a specific WhatsApp message.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **message** (`str, required`): Reply text
- **reply_to** (`str, required`): Message ID to reply to

**Returns:** Dictionary with success, message_id, to, replied_to

---

## `get_whatsapp_profile_tool`

Get WhatsApp Business profile information.

**Parameters:**

- **fields** (`str, optional`): Comma-separated fields to retrieve

**Returns:** Dictionary with success, profile data

---

## `send_whatsapp_contacts_tool`

Send contact cards via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **contacts** (`list, required`): List of contact dicts with:
- **name**: {first_name, last_name, formatted_name}
- **phones**: [{phone, type}]
- **emails**: [{email, type}] (optional)

**Returns:** Dictionary with success, message_id, to

---

## `send_whatsapp_interactive_tool`

Send interactive messages (buttons or lists) via WhatsApp.

**Parameters:**

- **to** (`str, required`): Recipient phone number
- **interactive_type** (`str, required`): 'button' or 'list'
- **header** (`str, optional`): Header text
- **body** (`str, required`): Body text
- **footer** (`str, optional`): Footer text
- **buttons** (`list, optional`): For button type [{id, title}] (max 3)
- **sections** (`list, optional`): For list type [{title, rows: [{id, title, description}]}]
- **button_text** (`str, optional`): Button text for list type

**Returns:** Dictionary with success, message_id, to

---

## `send_whatsapp_message`

Send WhatsApp message synchronously.

**Parameters:**

- **phone** (`str`)
- **message** (`str`)
- **provider** (`str`)

**Returns:** `Dict[str, Any]`

---

## `send_whatsapp_media`

Send WhatsApp media synchronously.

**Parameters:**

- **phone** (`str`)
- **media_path** (`str`)
- **caption** (`str`)
- **provider** (`str`)

**Returns:** `Dict[str, Any]`

---

## `get_whatsapp_status`

Get WhatsApp status synchronously.

**Parameters:**

- **provider** (`str`)

**Returns:** `Dict[str, Any]`
