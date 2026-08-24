/**
 * Shared utility functions for agentui components
 */

/**
 * Build an API URL with the configured base prefix.
 * When AgentUI is mounted at a sub-path (e.g. /agentui), all API calls
 * need to include that prefix. In standalone mode apiBase is empty.
 * @param {string} path - The API path (e.g. '/api/tools')
 * @returns {string} Full URL with prefix
 */
export function apiUrl(path) {
    const config = (typeof window !== 'undefined' && window.APP_CONFIG) || {};
    return (config.apiBase || '') + path;
}

/**
 * Format parameter label - convert snake_case to Title Case
 * @param {string} key - The parameter key
 * @returns {string} Formatted label
 */
export function formatParameterLabel(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Get CSS class for port type colors
 * @param {string} portType - The port type (image, detections)
 * @returns {string} CSS class name
 */
export function getPortTypeClass(portType) {
  const typeClasses = {
    'image': 'port-type-image',
    'detections': 'port-type-detections'
  };
  return typeClasses[portType] || 'port-type-image';
}

/**
 * Get parameter type from value
 * @param {*} value - The parameter value
 * @returns {string} Parameter type
 */
export function getParameterType(value) {
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number') return 'number';
  if (typeof value === 'string') {
    if (value.startsWith('#')) return 'color';
    if (value.includes('.jpg') || value.includes('.png') || value.includes('.jpeg')) return 'file';
    return 'text';
  }
  return 'text';
}

/**
 * Parse value according to type
 * @param {*} value - The value to parse
 * @param {string} type - The expected type
 * @returns {*} Parsed value
 */
export function parseValue(value, type) {
  if (type === 'number') return parseFloat(value) || 0;
  if (type === 'boolean') return value === 'true' || value === true;
  return value;
}

/**
 * Get category icon for node palette
 * @param {string} category - The category name
 * @returns {string} Emoji icon
 */
export function getCategoryIcon(category) {
  const icons = {
    'Input/Output': '📥',
    'Transform': '🔧',
    'Adjust': '🎨',
    'Filter': '🔍',
    'Analysis': '📊',
    'Detection': '🤖',
    'Combine': '🔀',
    'Other': '⚙️'
  };
  return icons[category] || '⚙️';
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