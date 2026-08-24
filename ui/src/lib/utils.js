/**
 * The small shared helpers the editor's components use.
 *
 * Deliberately thin. Anything that talks to the server lives in `api.js`, and anything about what
 * a parameter *is* comes from the catalogue the server derives -- the panel used to guess a widget
 * from the value it happened to hold, which cannot tell an unset number from a zero.
 */


/**
 * Format parameter label - convert snake_case to Title Case
 * @param {string} key - The parameter key
 * @returns {string} Formatted label
 */
export function formatParameterLabel(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}





/**
 * Convert text to CSS-safe class name (slugify)
 * Removes special characters, converts spaces to hyphens, lowercases
 * @param {string} text - The text to slugify
 * @returns {string} CSS-safe class name
 */
export function slugify(text) {
  if (!text) return '';
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '') // Remove special characters except spaces and hyphens
    .replace(/\s+/g, '-') // Convert spaces to hyphens
    .replace(/-+/g, '-') // Replace multiple hyphens with single
    .replace(/^-|-$/g, ''); // Remove leading/trailing hyphens
}

/**
 * Generate CSS classes for a node based on its type and category
 * @param {string} nodeType - The node type (e.g., 'ImageInput', 'Resize')
 * @param {string} category - The node category (e.g., 'Input/Output', 'Transform')
 * @returns {string} Space-separated CSS classes
 */
export function generateNodeClasses(nodeType, category) {
  const nodeNameClass = `node-${slugify(nodeType)}`;
  const nodeCategoryClass = `node-${slugify(category)}`;
  return `${nodeNameClass} ${nodeCategoryClass}`;
}