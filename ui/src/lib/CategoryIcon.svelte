<script>
  /**
   * The icon beside a node in the palette and on the canvas.
   *
   * A lookup rather than a chain of branches, so adding a category is one line and a category with
   * no icon is visibly the fallback rather than a silently wrong one. The keys are the categories
   * `mozo.workflow` actually declares; `tests/workflow/test_editor.py` is what keeps the two sets
   * from drifting apart.
   */
  export let category = '';
  export let size = 20;
  export let class_ = '';

  $: iconSize = `${size}px`;

  const ICONS = {
    // Input — a picture arriving
    Input: '<path d="M3 5h18v14H3z" stroke="currentColor" stroke-width="2"/><path d="M3 16l5-5 4 4 3-3 6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="8.5" cy="8.5" r="1.5" stroke="currentColor" stroke-width="2"/>',
    // Output — a file being written
    Output: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M7 10l5 5 5-5M12 15V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    // Detect — a box drawn around something
    Detect: '<path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/>',
    // Segment — a shape cut from the background
    Segment: '<path d="M4 20c2-8 6-12 14-16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M4 20h7a9 9 0 0 0 9-9V4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="3 3"/>',
    // Classify — naming, without locating
    Classify: '<path d="M20.6 13.4L12 4.8V2H5a3 3 0 0 0-3 3v7l8.6 8.6a2 2 0 0 0 2.8 0l7.2-7.2a2 2 0 0 0 0-2.8z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><circle cx="7" cy="7" r="1.5" fill="currentColor"/>',
    // Read — text
    Read: '<path d="M4 7V5h16v2M9 19h6M12 5v14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    // Depth — near and far
    Depth: '<path d="M12 3l9 5-9 5-9-5 9-5z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M3 13l9 5 9-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    // Transform — geometry
    Transform: '<path d="M21 2v6h-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M3 22v-6h6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    // Adjust — sliders
    Adjust: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    // Annotate — drawing on the image
    Annotate: '<path d="M12 19l7-7 3 3-7 7-3-3z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M2 2l7.586 7.586" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  };

  //: A circle, for a category nobody has drawn yet. Plain enough to be recognisable as "unstyled".
  const FALLBACK = '<circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="2"/>';

  $: paths = ICONS[category] || FALLBACK;
</script>

<svg class={class_} width={iconSize} height={iconSize} viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg">
  {@html paths}
</svg>

<style>
  svg {
    display: inline-block;
    color: #666;
    transition: color 0.15s ease;
  }
</style>
