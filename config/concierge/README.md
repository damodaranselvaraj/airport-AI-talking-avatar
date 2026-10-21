# Default concierge face

Put the image of the default concierge here: one clear, front-facing face,
as a `.png`, `.jpg` or `.jpeg` file (for example `concierge.png`). The first
image in this folder (sorted by name) is used. To use a file elsewhere, set
`CONCIERGE_IMAGE` in `.env`.

Requirements:

- You must have the right to use the image in the kiosk (your own photo,
  an illustration you made or licensed, or a licensed stock image). Do not
  use a photo of a real person without their consent.
- A single face, looking at the camera, with the mouth closed or relaxed, and
  roughly 512 px or more across the face works best.
- The server refuses to start with `AVATAR_ENABLED=true` if this folder has no
  image, so a missing concierge is caught at startup rather than mid-conversation.

Unlike `faces/`, this folder is committed on purpose: the concierge is part of
the kiosk, not a traveler's photo.
