Added a helper that normalizes RGBA PIL images into RGB `ImageClip`s with a broadcastable mask, then used it for the kinetic text (and related text clips) so MoviePy’s blit sees consistent formats.

Details:
- `kaggle/VideoAfisha/video_afisha.ipynb`: added `rgba_image_to_clip` to split RGB/alpha and expand the mask to `(H,W,1)` for safe compositing.
- `kaggle/VideoAfisha/video_afisha.ipynb`: switched `create_word_clip_fixed_height`, `create_single_word_clip`, and the sticker/bento clip creation to use the helper for consistent RGB output and masks.

Next steps:
1) Re-run the notebook from cell 6 onward to confirm the ValueError is gone.
2) Render a short scene that includes kinetic text to validate the composite behavior.