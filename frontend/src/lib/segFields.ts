// Per-method segmentation params: each is only relevant when the chosen --segment method
// uses it. Ported verbatim from the original app.js SEG_FIELD_METHODS so the form keeps
// hiding irrelevant knobs.
export const SEG_FIELD_METHODS: Record<string, string[]> = {
  grabcut_iters: ['grabcut'],
  grabcut_rect_margin: ['grabcut'],
  grabcut_min_object_area: ['grabcut'],
  otsu_blur_kernel: ['otsu'],
  otsu_threshold_adjustment: ['otsu'],
  morph_kernel: ['grabcut', 'otsu', 'threshold'],
  rembg_model: ['rembg'],
  sam_checkpoint: ['sam'],
  sam_model_type: ['sam'],
  sam_prompt_type: ['sam'],
  sam_points_per_side: ['sam'],
  sam_pred_iou_thresh: ['sam'],
  sam_stability_score_thresh: ['sam'],
}

/** A field is visible unless it's a per-method seg knob for a different method. */
export function isFieldVisible(name: string, segment: string): boolean {
  const methods = SEG_FIELD_METHODS[name]
  return !methods || methods.includes(segment)
}

// Which segmentation config knobs the tuner exposes for each method, in display order.
// These are rendered from the live /api/schema metadata (widget/help/default) but their
// values are kept LOCAL to the tuner (they don't write to the shared run config).
export const TUNER_FIELDS: Record<string, string[]> = {
  auto: [
    'seg_conf_thresh',
    'rembg_fallback',
    'rembg_provider',
    'strip_shadow',
    'shadow_sat_max',
    'bg_tolerance',
    'morph_kernel',
  ],
  threshold: ['bg_tolerance', 'morph_kernel'],
  otsu: ['otsu_blur_kernel', 'otsu_threshold_adjustment', 'morph_kernel'],
  grabcut: ['grabcut_iters', 'grabcut_rect_margin', 'grabcut_min_object_area', 'morph_kernel'],
  rembg: ['rembg_model', 'rembg_provider'],
  alpha: [],
  sam: [
    'sam_checkpoint',
    'sam_model_type',
    'sam_prompt_type',
    'sam_points_per_side',
    'sam_pred_iou_thresh',
    'sam_stability_score_thresh',
    'grabcut_rect_margin',
  ],
}
