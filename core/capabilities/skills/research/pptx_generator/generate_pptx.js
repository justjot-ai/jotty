#!/usr/bin/env node
/**
 * World-Class PowerPoint Generator for ArXiv Learning
 * McKinsey / Goldman Sachs / BCG / Apple Keynote inspired design
 *
 * Design Principles:
 * - Cinematic visual storytelling
 * - Precise typographic hierarchy
 * - Strategic negative space (40%+ whitespace)
 * - Data-driven visual hierarchy
 * - Subtle animations and transitions
 * - Consistent premium brand elements
 * - High contrast for readability
 *
 * Usage: node generate_pptx.js <json_file> <output_path>
 */

const PptxGenJS = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

// ============================================================================
// WORLD-CLASS DESIGN SYSTEM - PREMIUM TIER
// ============================================================================

const COLORS = {
    // Primary palette - Deep, authoritative (inspired by luxury brands)
    navy: '0a0f1a',           // True dark - maximum contrast
    navyMid: '141b2d',        // Mid navy - subtle backgrounds
    slate: '1e293b',          // Dark slate - body text
    steel: '475569',          // Steel gray - secondary text
    muted: '94a3b8',          // Muted gray - captions

    // Accent palette - Refined, premium feel
    gold: 'c9a227',           // Rich gold - premium accent (Rolex-inspired)
    goldLight: 'e5c76b',      // Light gold - highlights
    goldDark: '9a7b1a',       // Dark gold - shadows
    blue: '2563eb',           // Electric blue - links, emphasis
    blueLight: '60a5fa',      // Light blue - secondary accent
    blueDark: '1d4ed8',       // Deep blue - gradients

    // Semantic colors - Refined
    success: '059669',        // Emerald - positive/Bingo
    successLight: '34d399',   // Light emerald - badges
    successDark: '047857',    // Dark emerald - depth
    warning: 'd97706',        // Warm amber - caution
    danger: 'dc2626',         // Alert red

    // Backgrounds - Subtle gradation
    white: 'ffffff',
    offWhite: 'fafbfc',       // Barely-there gray
    cream: 'fefdfb',          // Warm white
    lightGray: 'f1f5f9',      // Cards, containers
    mediumGray: 'e2e8f0',     // Borders

    // Special
    codeBlock: '0d1117',      // GitHub dark - code background
    codeBorder: '30363d',     // Code border
    codeText: 'c9d1d9',       // Code text
    codeKeyword: 'ff7b72',    // Keywords
    codeString: 'a5d6ff',     // Strings
    codeComment: '8b949e'     // Comments
};

const FONTS = {
    // Premium font stack - uses system fonts for maximum compatibility
    heading: 'Segoe UI Light',     // Clean, modern headings (Windows)
    headingAlt: 'Calibri Light',   // Fallback
    body: 'Segoe UI',              // Readable body
    bodyAlt: 'Calibri',            // Fallback
    accent: 'Segoe UI Semibold',   // Numbers, stats, emphasis
    code: 'Cascadia Code',         // Modern monospace (fallback: Consolas)
    codeAlt: 'Consolas'
};

// Slide dimensions (16:9 cinematic)
const SLIDE = {
    width: 10,           // inches
    height: 5.625,       // inches
    marginX: 0.6,        // horizontal margin
    marginY: 0.5,        // vertical margin
    gutter: 0.25,        // space between elements
    contentWidth: 8.8,   // usable content width
    contentTop: 1.0,     // content start Y after header
    footerY: 5.35        // footer position
};

// Typography scale (modular scale 1.25)
const TYPE = {
    hero: 44,        // Title slide main
    h1: 32,          // Section headers
    h2: 24,          // Slide titles
    h3: 18,          // Subtitles
    body: 13,        // Body text
    small: 11,       // Captions
    tiny: 9,         // Footnotes
    micro: 7         // Legal
};

// Smart text fitting - estimates if text will overflow
const TEXT_UTILS = {
    // Approximate characters per inch at different font sizes
    charsPerInch: (fontSize) => 10 / (fontSize / 12),

    // Estimate lines needed for text at given width and font size
    estimateLines: (text, widthInches, fontSize) => {
        const cpi = TEXT_UTILS.charsPerInch(fontSize);
        const charsPerLine = Math.floor(widthInches * cpi);
        return Math.ceil(text.length / charsPerLine);
    },

    // Calculate optimal font size to fit text in area
    fitFontSize: (text, widthInches, heightInches, maxFont, minFont = 10) => {
        const lineHeight = 1.4; // line height multiplier
        for (let fs = maxFont; fs >= minFont; fs -= 1) {
            const lines = TEXT_UTILS.estimateLines(text, widthInches, fs);
            const totalHeight = (lines * fs * lineHeight) / 72; // convert to inches
            if (totalHeight <= heightInches) {
                return fs;
            }
        }
        return minFont;
    }
};

// Spacing system (8px base)
const SPACE = {
    xs: 0.05,   // 4px
    sm: 0.1,    // 8px
    md: 0.2,    // 16px
    lg: 0.35,   // 28px
    xl: 0.5,    // 40px
    xxl: 0.75   // 60px
};

// Animation presets (for future use)
const TRANSITIONS = {
    fade: { type: 'fade' },
    slideLeft: { type: 'push', direction: 'L' },
    slideUp: { type: 'push', direction: 'U' }
};

// ============================================================================
// VISUALIZATION ENGINE - World-Class Diagrams & Charts
// ============================================================================

/**
 * Premium diagram color palette - carefully curated for visual harmony
 */
const DIAGRAM_COLORS = {
    // Rich, saturated colors that work well together
    primary: ['1e3a5f', '2563eb', '3b82f6'],      // Deep blue spectrum
    secondary: ['065f46', '059669', '10b981'],    // Rich emerald spectrum
    tertiary: ['92400e', 'ea580c', 'f97316'],     // Warm orange spectrum
    quaternary: ['5b21b6', '7c3aed', '8b5cf6'],   // Royal purple spectrum

    // Neutral connectors
    connector: '64748b',
    connectorLight: '94a3b8',
    shadow: '1e293b'
};

/**
 * DiagramBuilder - World-Class Professional Diagrams
 *
 * Design principles:
 * - Clean, minimal aesthetic
 * - Proper visual hierarchy
 * - Consistent spacing (8px grid)
 * - Subtle depth through shadows
 * - Readable text at all sizes
 */
class DiagramBuilder {
    constructor(slide, startX, startY, width, height) {
        this.slide = slide;
        this.startX = startX;
        this.startY = startY;
        this.width = width;
        this.height = height;
    }

    /**
     * Draw a premium styled node with proper depth and typography
     */
    drawNode(x, y, w, h, label, options = {}) {
        const {
            color = DIAGRAM_COLORS.primary[1],
            textColor = 'ffffff',
            borderColor = null,
            subtitle = null,
            style = 'rounded',
            number = null
        } = options;

        const radius = style === 'pill' ? Math.min(w, h) / 2 :
                       style === 'rounded' ? 0.1 : 0;

        // Soft shadow (offset and blur effect via multiple layers)
        this.slide.addShape('roundRect', {
            x: x + 0.04, y: y + 0.05, w, h,
            fill: { color: DIAGRAM_COLORS.shadow, transparency: 75 },
            rectRadius: radius
        });

        // Main card
        this.slide.addShape('roundRect', {
            x, y, w, h,
            fill: { color },
            rectRadius: radius
        });

        // Subtle top highlight (glass effect)
        if (h > 0.5) {
            this.slide.addShape('roundRect', {
                x: x + 0.02, y: y + 0.02, w: w - 0.04, h: h * 0.35,
                fill: { color: 'ffffff', transparency: 88 },
                rectRadius: radius > 0 ? radius - 0.02 : 0
            });
        }

        // Calculate optimal font size based on text length and box size
        const maxCharsPerLine = Math.floor(w * 9);
        const needsWrap = label.length > maxCharsPerLine;
        const baseFontSize = Math.min(11, Math.max(8, w * 6.5));
        const fontSize = needsWrap ? baseFontSize * 0.9 : baseFontSize;

        // Label positioning
        const labelY = subtitle ? y + h * 0.12 : y + h * 0.1;
        const labelH = subtitle ? h * 0.52 : h * 0.8;

        this.slide.addText(label, {
            x: x + 0.08, y: labelY, w: w - 0.16, h: labelH,
            fontSize,
            bold: true,
            color: textColor,
            fontFace: FONTS.accent,
            align: 'center',
            valign: 'middle',
            wrap: true,
            lineSpacing: fontSize * 1.15
        });

        // Subtitle with proper spacing
        if (subtitle) {
            this.slide.addText(subtitle, {
                x: x + 0.08, y: y + h * 0.62, w: w - 0.16, h: h * 0.32,
                fontSize: Math.max(7, fontSize * 0.7),
                color: textColor,
                fontFace: FONTS.body,
                align: 'center',
                valign: 'top',
                transparency: 15
            });
        }

        // Optional number badge
        if (number !== null) {
            const badgeSize = 0.22;
            this.slide.addShape('ellipse', {
                x: x + w - badgeSize - 0.06, y: y + 0.06,
                w: badgeSize, h: badgeSize,
                fill: { color: 'ffffff', transparency: 20 }
            });
            this.slide.addText(String(number), {
                x: x + w - badgeSize - 0.06, y: y + 0.06,
                w: badgeSize, h: badgeSize,
                fontSize: 8, bold: true, color: textColor,
                fontFace: FONTS.accent, align: 'center', valign: 'middle'
            });
        }
    }

    /**
     * Draw a clean, modern arrow connector
     */
    drawArrow(x1, y1, x2, y2, options = {}) {
        const {
            color = DIAGRAM_COLORS.connector,
            thickness = 2.5,
            style = 'pointed'  // pointed, chevron
        } = options;

        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);

        if (length < 0.1) return;

        const angle = Math.atan2(dy, dx);
        const headLen = 0.1;

        // Draw line (shortened to make room for arrowhead)
        const lineEndX = x2 - headLen * 1.2 * Math.cos(angle);
        const lineEndY = y2 - headLen * 1.2 * Math.sin(angle);

        this.slide.addShape('line', {
            x: x1, y: y1,
            w: Math.sqrt((lineEndX - x1) ** 2 + (lineEndY - y1) ** 2),
            h: 0,
            line: { color, pt: thickness },
            rotate: angle * 180 / Math.PI
        });

        // Modern chevron arrowhead
        const headAngle = Math.PI / 6;  // 30 degrees
        const p1x = x2 - headLen * Math.cos(angle - headAngle);
        const p1y = y2 - headLen * Math.sin(angle - headAngle);
        const p2x = x2 - headLen * Math.cos(angle + headAngle);
        const p2y = y2 - headLen * Math.sin(angle + headAngle);

        // Left side of chevron
        this.slide.addShape('line', {
            x: p1x, y: p1y,
            w: Math.sqrt((x2 - p1x) ** 2 + (y2 - p1y) ** 2),
            h: 0,
            line: { color, pt: thickness + 0.5 },
            rotate: Math.atan2(y2 - p1y, x2 - p1x) * 180 / Math.PI
        });

        // Right side of chevron
        this.slide.addShape('line', {
            x: p2x, y: p2y,
            w: Math.sqrt((x2 - p2x) ** 2 + (y2 - p2y) ** 2),
            h: 0,
            line: { color, pt: thickness + 0.5 },
            rotate: Math.atan2(y2 - p2y, x2 - p2x) * 180 / Math.PI
        });
    }

    /**
     * Draw a vertical arrow (for architecture diagrams)
     */
    drawVerticalConnector(x, y1, y2, options = {}) {
        const { color = DIAGRAM_COLORS.connectorLight, width = 0.5 } = options;
        const midX = x;
        const height = y2 - y1;

        // Vertical line
        this.slide.addShape('rect', {
            x: midX - 0.015, y: y1,
            w: 0.03, h: height - 0.12,
            fill: { color }
        });

        // Down arrow
        this.slide.addShape('triangle', {
            x: midX - 0.08, y: y2 - 0.14,
            w: 0.16, h: 0.12,
            fill: { color },
            rotate: 180
        });
    }

    /**
     * Draw a premium flow diagram (horizontal process)
     * AUTO-SCALES nodes to fit within available width
     */
    drawFlowDiagram(nodes, options = {}) {
        // Limit nodes to prevent overflow (max 5)
        const limitedNodes = nodes.slice(0, 5);
        const nodeCount = limitedNodes.length;

        // Calculate optimal sizes to fit available width
        const availableWidth = this.width - 0.2;
        const minGap = 0.2;
        const minNodeWidth = 1.2;

        // Calculate node width and gap dynamically
        let gap = options.gap || 0.35;
        let nodeWidth = options.nodeWidth || 2.1;

        // Total needed = nodeCount * nodeWidth + (nodeCount-1) * gap
        let totalNeeded = nodeCount * nodeWidth + (nodeCount - 1) * gap;

        if (totalNeeded > availableWidth) {
            // Scale down proportionally
            const scale = availableWidth / totalNeeded;
            nodeWidth = Math.max(minNodeWidth, nodeWidth * scale);
            gap = Math.max(minGap, gap * scale);
        }

        // Recalculate with final values
        const totalWidth = nodeCount * nodeWidth + (nodeCount - 1) * gap;
        const offsetX = this.startX + (this.width - totalWidth) / 2;

        // Scale node height based on available space
        let nodeHeight = Math.min(options.nodeHeight || 1.0, this.height - 0.3);
        const y = this.startY + (this.height - nodeHeight) / 2;

        const colorSets = [
            DIAGRAM_COLORS.primary,
            DIAGRAM_COLORS.secondary,
            DIAGRAM_COLORS.tertiary,
            DIAGRAM_COLORS.quaternary
        ];

        limitedNodes.forEach((node, i) => {
            const x = offsetX + i * (nodeWidth + gap);
            const colors = colorSets[i % colorSets.length];

            this.drawNode(x, y, nodeWidth, nodeHeight, node.label, {
                color: colors[1],
                subtitle: node.subtitle,
                style: 'rounded',
                number: i + 1
            });

            // Arrow to next node (only if there's room)
            if (i < limitedNodes.length - 1 && gap >= minGap) {
                const arrowY = y + nodeHeight / 2;
                this.drawArrow(
                    x + nodeWidth + 0.05,
                    arrowY,
                    x + nodeWidth + gap - 0.05,
                    arrowY,
                    { color: colors[1] }
                );
            }
        });
    }

    /**
     * Draw a premium architecture diagram (vertical layers)
     * AUTO-SCALES to fit within bounds - prevents overflow
     */
    drawArchitectureDiagram(layers, options = {}) {
        // Limit layers to fit within available height
        const maxLayers = Math.min(layers.length, 6);
        const limitedLayers = layers.slice(0, maxLayers);

        // Calculate optimal sizes based on available space
        const availableHeight = this.height - 0.1;  // Small margin
        const totalGaps = (limitedLayers.length - 1);

        // Dynamic sizing: shrink to fit
        let boxHeight = options.boxHeight || 0.65;
        let layerGap = options.layerGap || 0.35;
        const minBoxHeight = 0.4;
        const minGap = 0.15;

        // Calculate total height needed and scale down if necessary
        let totalNeeded = limitedLayers.length * boxHeight + totalGaps * layerGap;
        if (totalNeeded > availableHeight) {
            const scale = availableHeight / totalNeeded;
            boxHeight = Math.max(minBoxHeight, boxHeight * scale);
            layerGap = Math.max(minGap, layerGap * scale);
        }

        // Find max boxes in any layer to calculate box width
        const maxBoxesInLayer = Math.min(
            Math.max(...limitedLayers.map(l => (Array.isArray(l) ? l : (l.items || [l])).length)),
            5  // Limit to 5 boxes per row max
        );
        const boxGap = options.boxGap || 0.15;
        let boxWidth = options.boxWidth || 1.45;

        // Scale box width to fit available width
        const availableWidth = this.width - 0.2;
        const totalBoxWidth = maxBoxesInLayer * boxWidth + (maxBoxesInLayer - 1) * boxGap;
        if (totalBoxWidth > availableWidth) {
            boxWidth = (availableWidth - (maxBoxesInLayer - 1) * boxGap) / maxBoxesInLayer;
        }

        let currentY = this.startY;
        const colorSets = [
            DIAGRAM_COLORS.primary,
            DIAGRAM_COLORS.secondary,
            DIAGRAM_COLORS.tertiary,
            DIAGRAM_COLORS.quaternary
        ];

        limitedLayers.forEach((layer, layerIdx) => {
            const boxes = Array.isArray(layer) ? layer : [layer];
            let boxItems = layer.items || boxes;
            // Limit boxes per row
            boxItems = boxItems.slice(0, 5);
            const numBoxes = boxItems.length;
            const totalWidth = numBoxes * boxWidth + (numBoxes - 1) * boxGap;
            const offsetX = this.startX + (this.width - totalWidth) / 2;
            const colorSet = colorSets[layerIdx % colorSets.length];

            boxItems.forEach((box, boxIdx) => {
                const x = offsetX + boxIdx * (boxWidth + boxGap);
                const label = typeof box === 'string' ? box : box.label;
                const subtitle = typeof box === 'object' ? box.subtitle : null;

                this.drawNode(x, currentY, boxWidth, boxHeight, label, {
                    color: colorSet[1],
                    subtitle,
                    style: 'rounded'
                });
            });

            // Connector to next layer
            if (layerIdx < limitedLayers.length - 1) {
                const centerX = this.startX + this.width / 2;
                this.drawVerticalConnector(
                    centerX,
                    currentY + boxHeight + 0.03,
                    currentY + boxHeight + layerGap - 0.03,
                    { color: colorSet[1] }
                );
            }

            currentY += boxHeight + layerGap;
        });
    }

    /**
     * Draw premium concept map (hub and spoke with clean connections)
     * Uses a cleaner grid-like layout to avoid overlapping lines
     */
    drawConceptMap(centerNode, relatedNodes, options = {}) {
        // Limit to 4 nodes for cleanest layout (no overlapping lines)
        const limitedNodes = relatedNodes.slice(0, 4);
        const nodeCount = limitedNodes.length;

        // Calculate sizes based on available space
        const centerSize = Math.min(2.2, this.width * 0.22);
        const centerH = centerSize * 0.45;
        const nodeSize = Math.min(1.6, this.width * 0.18);
        const nodeH = nodeSize * 0.45;

        const centerX = this.startX + this.width / 2;
        const centerY = this.startY + this.height / 2;

        // Draw center node first (so lines go behind it)
        this.drawNode(
            centerX - centerSize / 2,
            centerY - centerH / 2,
            centerSize,
            centerH,
            centerNode,
            {
                color: DIAGRAM_COLORS.primary[1],
                style: 'rounded'
            }
        );

        const colorSets = [
            DIAGRAM_COLORS.secondary,
            DIAGRAM_COLORS.tertiary,
            DIAGRAM_COLORS.quaternary,
            DIAGRAM_COLORS.primary
        ];

        // Position nodes in cardinal directions (no overlapping lines)
        // Top, Right, Bottom, Left - clean non-crossing layout
        const positions = [
            { dx: 0, dy: -1.6, lineDir: 'vertical' },    // Top
            { dx: 2.8, dy: 0, lineDir: 'horizontal' },   // Right
            { dx: 0, dy: 1.6, lineDir: 'vertical' },     // Bottom
            { dx: -2.8, dy: 0, lineDir: 'horizontal' },  // Left
        ];

        limitedNodes.forEach((node, i) => {
            const pos = positions[i];
            const colorSet = colorSets[i % colorSets.length];

            // Calculate node position
            let nodeX = centerX + pos.dx - nodeSize / 2;
            let nodeY = centerY + pos.dy - nodeH / 2;

            // Clamp to bounds
            nodeX = Math.max(this.startX + 0.1, Math.min(nodeX, this.startX + this.width - nodeSize - 0.1));
            nodeY = Math.max(this.startY + 0.1, Math.min(nodeY, this.startY + this.height - nodeH - 0.1));

            // Draw connection line (straight, non-overlapping)
            if (pos.lineDir === 'vertical') {
                // Vertical line from center to node
                const lineX = centerX;
                const lineStartY = pos.dy < 0 ? centerY - centerH / 2 - 0.05 : centerY + centerH / 2 + 0.05;
                const lineEndY = pos.dy < 0 ? nodeY + nodeH + 0.05 : nodeY - 0.05;
                const lineH = Math.abs(lineEndY - lineStartY);

                this.slide.addShape('rect', {
                    x: lineX - 0.02,
                    y: Math.min(lineStartY, lineEndY),
                    w: 0.04,
                    h: lineH,
                    fill: { color: colorSet[2] }
                });

                // Arrow head
                const arrowY = pos.dy < 0 ? lineEndY + 0.08 : lineEndY - 0.08;
                this.slide.addShape('triangle', {
                    x: lineX - 0.06,
                    y: arrowY,
                    w: 0.12,
                    h: 0.08,
                    fill: { color: colorSet[2] },
                    rotate: pos.dy < 0 ? 0 : 180
                });
            } else {
                // Horizontal line from center to node
                const lineY = centerY;
                const lineStartX = pos.dx < 0 ? centerX - centerSize / 2 - 0.05 : centerX + centerSize / 2 + 0.05;
                const lineEndX = pos.dx < 0 ? nodeX + nodeSize + 0.05 : nodeX - 0.05;
                const lineW = Math.abs(lineEndX - lineStartX);

                this.slide.addShape('rect', {
                    x: Math.min(lineStartX, lineEndX),
                    y: lineY - 0.02,
                    w: lineW,
                    h: 0.04,
                    fill: { color: colorSet[2] }
                });

                // Arrow head
                const arrowX = pos.dx < 0 ? lineEndX + 0.08 : lineEndX - 0.08;
                this.slide.addShape('triangle', {
                    x: arrowX,
                    y: lineY - 0.06,
                    w: 0.08,
                    h: 0.12,
                    fill: { color: colorSet[2] },
                    rotate: pos.dx < 0 ? 90 : 270
                });
            }

            // Draw node
            const label = typeof node === 'string' ? node : node.label;
            this.drawNode(nodeX, nodeY, nodeSize, nodeH, label, {
                color: colorSet[1],
                style: 'rounded'
            });
        });
    }

    /**
     * Draw a simple bar chart
     * AUTO-SCALES to fit within bounds
     */
    drawBarChart(data, options = {}) {
        // Limit bars to prevent overflow (max 6)
        const limitedData = data.slice(0, 6);
        const barCount = limitedData.length;

        // Calculate optimal bar width and gap
        const availableWidth = this.width - 0.3;
        let barWidth = options.barWidth || 0.6;
        let gap = options.gap || 0.3;

        // Scale down if needed
        const totalNeeded = barCount * barWidth + (barCount - 1) * gap;
        if (totalNeeded > availableWidth) {
            const scale = availableWidth / totalNeeded;
            barWidth = Math.max(0.35, barWidth * scale);
            gap = Math.max(0.15, gap * scale);
        }

        const maxBarHeight = Math.min(options.maxBarHeight || 2.0, this.height - 0.6);
        const showValues = options.showValues !== false;

        const maxValue = Math.max(...limitedData.map(d => d.value));
        const totalWidth = barCount * barWidth + (barCount - 1) * gap;
        const offsetX = this.startX + (this.width - totalWidth) / 2;
        const baseY = this.startY + this.height - 0.35;

        const colors = [
            DIAGRAM_COLORS.primary[1],
            DIAGRAM_COLORS.secondary[1],
            DIAGRAM_COLORS.tertiary[1],
            DIAGRAM_COLORS.quaternary[1]
        ];

        // Scale label font based on bar width
        const labelFontSize = barWidth > 0.5 ? 8 : 7;
        const valueFontSize = barWidth > 0.5 ? 10 : 8;

        limitedData.forEach((item, i) => {
            const x = offsetX + i * (barWidth + gap);
            const barHeight = (item.value / maxValue) * maxBarHeight;
            const y = baseY - barHeight;
            const color = colors[i % colors.length];

            // Bar shadow
            this.slide.addShape('rect', {
                x: x + 0.015, y: y + 0.015, w: barWidth, h: barHeight,
                fill: { color: '000000', transparency: 85 },
                rectRadius: 0.03
            });

            // Bar
            this.slide.addShape('rect', {
                x, y, w: barWidth, h: barHeight,
                fill: { color },
                rectRadius: 0.03
            });

            // Value label
            if (showValues) {
                this.slide.addText(String(item.value), {
                    x, y: y - 0.2, w: barWidth, h: 0.18,
                    fontSize: valueFontSize, bold: true, color: COLORS.slate,
                    fontFace: FONTS.accent, align: 'center'
                });
            }

            // Category label - truncate if needed
            const maxLabelChars = Math.floor(barWidth * 12);
            const displayLabel = item.label.length > maxLabelChars
                ? item.label.slice(0, maxLabelChars - 1) + '…'
                : item.label;
            this.slide.addText(displayLabel, {
                x: x - 0.05, y: baseY + 0.03, w: barWidth + 0.1, h: 0.25,
                fontSize: labelFontSize, color: COLORS.steel,
                fontFace: FONTS.body, align: 'center'
            });
        });
    }

    /**
     * Draw premium comparison diagram (side by side)
     * AUTO-SCALES bullet points to fit within bounds
     */
    drawComparison(left, right, options = {}) {
        const { vsLabel = 'vs' } = options;
        const gap = 0.7;
        const boxW = (this.width - gap) / 2;
        const boxH = this.height - 0.1;
        const headerH = 0.5;

        // Limit points to fit within box height
        const availableContentH = boxH - headerH - 0.15;
        const maxPoints = Math.min(
            Math.max(left.points.length, right.points.length),
            Math.floor(availableContentH / 0.42)  // Minimum row height
        );
        const leftPoints = left.points.slice(0, maxPoints);
        const rightPoints = right.points.slice(0, maxPoints);

        // Calculate dynamic row height
        const rowHeight = Math.min(0.55, availableContentH / maxPoints);
        const fontSize = rowHeight > 0.45 ? 11 : (rowHeight > 0.38 ? 10 : 9);
        const iconSize = fontSize + 2;

        // Left card (Before/Traditional - warm color)
        this.slide.addShape('roundRect', {
            x: this.startX + 0.03, y: this.startY + 0.04, w: boxW, h: boxH,
            fill: { color: DIAGRAM_COLORS.shadow, transparency: 80 },
            rectRadius: 0.12
        });
        this.slide.addShape('roundRect', {
            x: this.startX, y: this.startY, w: boxW, h: boxH,
            fill: { color: 'fef3e2' },
            line: { color: DIAGRAM_COLORS.tertiary[1], pt: 2 },
            rectRadius: 0.12
        });
        // Left header bar
        this.slide.addShape('roundRect', {
            x: this.startX, y: this.startY, w: boxW, h: headerH,
            fill: { color: DIAGRAM_COLORS.tertiary[1] },
            rectRadius: 0.12
        });
        this.slide.addShape('rect', {
            x: this.startX, y: this.startY + headerH - 0.12, w: boxW, h: 0.12,
            fill: { color: DIAGRAM_COLORS.tertiary[1] }
        });
        this.slide.addText(left.title, {
            x: this.startX, y: this.startY + 0.08, w: boxW, h: headerH - 0.15,
            fontSize: 14, bold: true, color: 'ffffff',
            fontFace: FONTS.accent, align: 'center', valign: 'middle'
        });
        // Left bullet points - dynamic spacing
        leftPoints.forEach((point, i) => {
            const y = this.startY + headerH + 0.12 + i * rowHeight;
            this.slide.addText('✗', {
                x: this.startX + 0.15, y, w: 0.25, h: rowHeight - 0.05,
                fontSize: iconSize, color: DIAGRAM_COLORS.tertiary[1],
                fontFace: FONTS.body, valign: 'middle'
            });
            // Truncate long points
            const maxChars = Math.floor((boxW - 0.6) * 8);
            const displayPoint = point.length > maxChars ? point.slice(0, maxChars - 1) + '…' : point;
            this.slide.addText(displayPoint, {
                x: this.startX + 0.42, y, w: boxW - 0.55, h: rowHeight - 0.02,
                fontSize, color: COLORS.slate,
                fontFace: FONTS.body, valign: 'middle'
            });
        });

        // VS badge
        const vsX = this.startX + boxW + (gap - 0.45) / 2;
        const vsY = this.startY + boxH / 2 - 0.22;
        this.slide.addShape('ellipse', {
            x: vsX + 0.02, y: vsY + 0.03, w: 0.45, h: 0.45,
            fill: { color: DIAGRAM_COLORS.shadow, transparency: 70 }
        });
        this.slide.addShape('ellipse', {
            x: vsX, y: vsY, w: 0.45, h: 0.45,
            fill: { color: COLORS.navy }
        });
        this.slide.addText(vsLabel, {
            x: vsX, y: vsY, w: 0.45, h: 0.45,
            fontSize: 11, bold: true, color: COLORS.white,
            fontFace: FONTS.accent, align: 'center', valign: 'middle'
        });

        // Right card (After/New - cool color)
        const rightX = this.startX + boxW + gap;
        this.slide.addShape('roundRect', {
            x: rightX + 0.03, y: this.startY + 0.04, w: boxW, h: boxH,
            fill: { color: DIAGRAM_COLORS.shadow, transparency: 80 },
            rectRadius: 0.12
        });
        this.slide.addShape('roundRect', {
            x: rightX, y: this.startY, w: boxW, h: boxH,
            fill: { color: 'e6f7f1' },
            line: { color: DIAGRAM_COLORS.secondary[1], pt: 2 },
            rectRadius: 0.12
        });
        // Right header bar
        this.slide.addShape('roundRect', {
            x: rightX, y: this.startY, w: boxW, h: headerH,
            fill: { color: DIAGRAM_COLORS.secondary[1] },
            rectRadius: 0.12
        });
        this.slide.addShape('rect', {
            x: rightX, y: this.startY + headerH - 0.12, w: boxW, h: 0.12,
            fill: { color: DIAGRAM_COLORS.secondary[1] }
        });
        this.slide.addText(right.title, {
            x: rightX, y: this.startY + 0.08, w: boxW, h: headerH - 0.15,
            fontSize: 14, bold: true, color: 'ffffff',
            fontFace: FONTS.accent, align: 'center', valign: 'middle'
        });
        // Right bullet points - dynamic spacing
        rightPoints.forEach((point, i) => {
            const y = this.startY + headerH + 0.12 + i * rowHeight;
            this.slide.addText('✓', {
                x: rightX + 0.15, y, w: 0.25, h: rowHeight - 0.05,
                fontSize: iconSize, color: DIAGRAM_COLORS.secondary[1],
                fontFace: FONTS.body, valign: 'middle'
            });
            // Truncate long points
            const maxChars = Math.floor((boxW - 0.6) * 8);
            const displayPoint = point.length > maxChars ? point.slice(0, maxChars - 1) + '…' : point;
            this.slide.addText(displayPoint, {
                x: rightX + 0.42, y, w: boxW - 0.55, h: rowHeight - 0.02,
                fontSize, color: COLORS.slate,
                fontFace: FONTS.body, valign: 'middle'
            });
        });
    }

    /**
     * Draw premium timeline with connected nodes
     * AUTO-SCALES to fit within bounds
     */
    drawTimeline(events, options = {}) {
        // Limit events to prevent overflow (max 5)
        const limitedEvents = events.slice(0, 5);
        const eventCount = limitedEvents.length;

        const lineY = this.startY + this.height / 2;
        const nodeRadius = eventCount > 4 ? 0.14 : 0.18;
        const eventSpacing = (this.width - 0.5) / (eventCount - 1 || 1);

        // Main timeline track (gradient effect with multiple lines)
        this.slide.addShape('roundRect', {
            x: this.startX + 0.1, y: lineY - 0.035, w: this.width - 0.2, h: 0.07,
            fill: { color: DIAGRAM_COLORS.connectorLight },
            rectRadius: 0.035
        });

        const colorSets = [
            DIAGRAM_COLORS.primary,
            DIAGRAM_COLORS.secondary,
            DIAGRAM_COLORS.tertiary,
            DIAGRAM_COLORS.quaternary
        ];

        // Scale label card width based on event count
        const labelW = eventCount > 4 ? 0.9 : 1.1;
        const labelH = eventCount > 4 ? 0.4 : 0.5;
        const connectorH = eventCount > 4 ? 0.28 : 0.35;

        limitedEvents.forEach((event, i) => {
            const x = this.startX + 0.25 + i * eventSpacing;
            const colorSet = colorSets[i % colorSets.length];
            const isTop = i % 2 === 0;

            // Connector line from track to label
            const connectorY = isTop ? lineY - nodeRadius - connectorH : lineY + nodeRadius + 0.02;
            this.slide.addShape('rect', {
                x: x - 0.012, y: isTop ? connectorY : lineY + nodeRadius,
                w: 0.024, h: connectorH,
                fill: { color: colorSet[2] }
            });

            // Node shadow
            this.slide.addShape('ellipse', {
                x: x - nodeRadius + 0.015, y: lineY - nodeRadius + 0.02,
                w: nodeRadius * 2, h: nodeRadius * 2,
                fill: { color: DIAGRAM_COLORS.shadow, transparency: 75 }
            });

            // Node circle
            this.slide.addShape('ellipse', {
                x: x - nodeRadius, y: lineY - nodeRadius,
                w: nodeRadius * 2, h: nodeRadius * 2,
                fill: { color: colorSet[1] }
            });

            // Inner highlight
            this.slide.addShape('ellipse', {
                x: x - nodeRadius + 0.03, y: lineY - nodeRadius + 0.03,
                w: nodeRadius * 1.1, h: nodeRadius * 1.1,
                fill: { color: 'ffffff', transparency: 85 }
            });

            // Label card - scaled based on event count
            const labelY = isTop ? lineY - 0.95 : lineY + 0.4;
            this.slide.addShape('roundRect', {
                x: x - labelW / 2, y: labelY, w: labelW, h: labelH,
                fill: { color: colorSet[2], transparency: 60 },
                rectRadius: 0.05
            });

            // Truncate label if needed
            const maxLabelChars = Math.floor(labelW * 10);
            const displayLabel = event.label.length > maxLabelChars
                ? event.label.slice(0, maxLabelChars - 1) + '…'
                : event.label;

            this.slide.addText(displayLabel, {
                x: x - labelW / 2, y: labelY + 0.03, w: labelW, h: labelH * 0.55,
                fontSize: eventCount > 4 ? 8 : 10, bold: true, color: colorSet[0],
                fontFace: FONTS.accent, align: 'center', valign: 'middle'
            });

            if (event.subtitle) {
                this.slide.addText(event.subtitle, {
                    x: x - labelW / 2, y: labelY + labelH * 0.55, w: labelW, h: labelH * 0.4,
                    fontSize: eventCount > 4 ? 6.5 : 8, color: COLORS.steel,
                    fontFace: FONTS.body, align: 'center', valign: 'top'
                });
            }
        });
    }

    /**
     * Draw premium metric cards with visual hierarchy
     * AUTO-SCALES to fit within bounds
     */
    drawMetricCards(metrics, options = {}) {
        // Limit metrics to prevent overflow (max 4)
        const limitedMetrics = metrics.slice(0, 4);
        const metricCount = limitedMetrics.length;

        const gap = metricCount > 3 ? 0.18 : 0.25;
        const cardW = (this.width - gap * (metricCount - 1)) / metricCount;
        const cardH = Math.min(this.height, 1.8);  // Cap card height

        const colorSets = [
            DIAGRAM_COLORS.primary,
            DIAGRAM_COLORS.secondary,
            DIAGRAM_COLORS.tertiary,
            DIAGRAM_COLORS.quaternary
        ];

        // Scale font sizes based on card width
        const valueFontSize = cardW > 2 ? 32 : (cardW > 1.5 ? 26 : 22);
        const labelFontSize = cardW > 2 ? 11 : (cardW > 1.5 ? 10 : 9);

        limitedMetrics.forEach((metric, i) => {
            const x = this.startX + i * (cardW + gap);
            const colorSet = colorSets[i % colorSets.length];

            // Card shadow
            this.slide.addShape('roundRect', {
                x: x + 0.02, y: this.startY + 0.03, w: cardW, h: cardH,
                fill: { color: DIAGRAM_COLORS.shadow, transparency: 80 },
                rectRadius: 0.08
            });

            // Card background
            this.slide.addShape('roundRect', {
                x, y: this.startY, w: cardW, h: cardH,
                fill: { color: 'ffffff' },
                line: { color: colorSet[2], pt: 1.5 },
                rectRadius: 0.08
            });

            // Top accent bar
            this.slide.addShape('roundRect', {
                x, y: this.startY, w: cardW, h: 0.07,
                fill: { color: colorSet[1] },
                rectRadius: 0.08
            });
            this.slide.addShape('rect', {
                x, y: this.startY + 0.05, w: cardW, h: 0.02,
                fill: { color: colorSet[1] }
            });

            // Large value - scaled font
            this.slide.addText(metric.value, {
                x, y: this.startY + 0.15, w: cardW, h: cardH * 0.45,
                fontSize: valueFontSize, bold: true, color: colorSet[0],
                fontFace: FONTS.accent, align: 'center', valign: 'middle'
            });

            // Label - scaled font
            this.slide.addText(metric.label, {
                x, y: this.startY + cardH * 0.55, w: cardW, h: cardH * 0.2,
                fontSize: labelFontSize, bold: true, color: COLORS.slate,
                fontFace: FONTS.accent, align: 'center', valign: 'top'
            });

            // Subtitle
            if (metric.subtitle) {
                this.slide.addText(metric.subtitle, {
                    x: x + 0.05, y: this.startY + cardH * 0.72, w: cardW - 0.1, h: cardH * 0.25,
                    fontSize: labelFontSize - 2, color: COLORS.muted,
                    fontFace: FONTS.body, align: 'center', valign: 'top'
                });
            }
        });
    }
}

// ============================================================================
// PRESENTATION GENERATOR CLASS
// ============================================================================

class WorldClassPresentation {
    constructor(data) {
        this.data = data;
        this.pptx = new PptxGenJS();
        this.slideNumber = 0;
        this.initialize();
    }

    initialize() {
        // Metadata
        this.pptx.author = 'Jotty AI';
        this.pptx.title = this.data.paper_title;
        this.pptx.subject = `Research Learning: ${this.data.arxiv_id}`;
        this.pptx.company = 'Jotty AI Research';
        this.pptx.revision = '1';

        // 16:9 widescreen
        this.pptx.layout = 'LAYOUT_16x9';

        // Define master slides for consistency
        this.defineMasters();
    }

    defineMasters() {
        // Title Slide Master - Cinematic, dark, premium
        this.pptx.defineSlideMaster({
            title: 'MASTER_TITLE',
            background: { color: COLORS.navy },
            objects: [
                // Gradient-like layered background effect
                { rect: { x: 0, y: 4.8, w: '100%', h: 0.85, fill: { color: COLORS.navyMid } } },
                // Premium gold accent bar at bottom
                { rect: { x: 0, y: 5.4, w: '100%', h: 0.225, fill: { color: COLORS.gold } } },
                // Subtle top accent - thin gold line
                { rect: { x: SLIDE.marginX, y: 0.35, w: 2.5, h: 0.015, fill: { color: COLORS.gold } } },
                // Decorative corner accent
                { rect: { x: 9.2, y: 0.3, w: 0.4, h: 0.015, fill: { color: COLORS.goldDark } } },
                { rect: { x: 9.55, y: 0.3, w: 0.015, h: 0.4, fill: { color: COLORS.goldDark } } }
            ]
        });

        // Section Divider Master - Bold, impactful
        this.pptx.defineSlideMaster({
            title: 'MASTER_SECTION',
            background: { color: COLORS.blueDark },
            objects: [
                // Gradient layer
                { rect: { x: 0, y: 4.5, w: '100%', h: 1.125, fill: { color: COLORS.blue } } },
                // Gold accent bar
                { rect: { x: 0, y: 5.4, w: '100%', h: 0.225, fill: { color: COLORS.gold } } },
                // Large decorative number watermark position (text added per slide)
            ]
        });

        // Content Slide Master - Clean, professional
        this.pptx.defineSlideMaster({
            title: 'MASTER_CONTENT',
            background: { color: COLORS.white },
            objects: [
                // Header bar with subtle gradient effect
                { rect: { x: 0, y: 0, w: '100%', h: 0.55, fill: { color: COLORS.navy } } },
                { rect: { x: 0, y: 0.55, w: '100%', h: 0.05, fill: { color: COLORS.navyMid } } },
                // Gold accent line under header
                { rect: { x: 0, y: 0.6, w: '100%', h: 0.025, fill: { color: COLORS.gold } } },
                // Footer bar
                { rect: { x: 0, y: SLIDE.footerY, w: '100%', h: 0.275, fill: { color: COLORS.offWhite } } },
                // Footer top border
                { rect: { x: 0, y: SLIDE.footerY, w: '100%', h: 0.01, fill: { color: COLORS.mediumGray } } }
            ]
        });

        // Insight Slide Master - For key "Bingo" moments
        this.pptx.defineSlideMaster({
            title: 'MASTER_INSIGHT',
            background: { color: COLORS.cream },
            objects: [
                // Header bar - success green
                { rect: { x: 0, y: 0, w: '100%', h: 0.55, fill: { color: COLORS.successDark } } },
                { rect: { x: 0, y: 0.55, w: '100%', h: 0.05, fill: { color: COLORS.success } } },
                // Gold accent line
                { rect: { x: 0, y: 0.6, w: '100%', h: 0.025, fill: { color: COLORS.gold } } },
                // Subtle side accent
                { rect: { x: 0, y: 0.6, w: 0.08, h: 4.75, fill: { color: COLORS.successLight } } },
                // Footer
                { rect: { x: 0, y: SLIDE.footerY, w: '100%', h: 0.275, fill: { color: COLORS.offWhite } } },
                { rect: { x: 0, y: SLIDE.footerY, w: '100%', h: 0.01, fill: { color: COLORS.mediumGray } } }
            ]
        });

        // Quote/Highlight Slide Master - For impactful statements
        this.pptx.defineSlideMaster({
            title: 'MASTER_QUOTE',
            background: { color: COLORS.navy },
            objects: [
                // Large quotation mark watermark area
                { rect: { x: 0, y: 4.8, w: '100%', h: 0.625, fill: { color: COLORS.navyMid } } },
                { rect: { x: 0, y: 5.4, w: '100%', h: 0.225, fill: { color: COLORS.gold } } }
            ]
        });

        // Data/Metrics Slide Master - For statistics
        this.pptx.defineSlideMaster({
            title: 'MASTER_DATA',
            background: { color: COLORS.white },
            objects: [
                { rect: { x: 0, y: 0, w: '100%', h: 0.55, fill: { color: COLORS.navy } } },
                { rect: { x: 0, y: 0.55, w: '100%', h: 0.05, fill: { color: COLORS.navyMid } } },
                { rect: { x: 0, y: 0.6, w: '100%', h: 0.025, fill: { color: COLORS.blue } } },
                { rect: { x: 0, y: SLIDE.footerY, w: '100%', h: 0.275, fill: { color: COLORS.offWhite } } },
                { rect: { x: 0, y: SLIDE.footerY, w: '100%', h: 0.01, fill: { color: COLORS.mediumGray } } }
            ]
        });
    }

    // ========================================================================
    // SLIDE GENERATORS
    // ========================================================================

    addTitleSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_TITLE' });
        this.slideNumber++;

        // Decorative large watermark number
        slide.addText('01', {
            x: 6.5, y: 0.8, w: 3.5, h: 2,
            fontSize: 120, color: COLORS.navyMid,
            fontFace: FONTS.heading, align: 'right', valign: 'top',
            transparency: 70
        });

        // Premium label with refined typography
        slide.addText('RESEARCH LEARNING', {
            x: SLIDE.marginX, y: 0.55, w: 4, h: 0.28,
            fontSize: TYPE.tiny, bold: true, color: COLORS.goldLight,
            fontFace: FONTS.accent, charSpacing: 4
        });

        // Main title - cinematic, commanding presence
        const title = this.wrapText(this.data.paper_title, 42);
        slide.addText(title, {
            x: SLIDE.marginX, y: 1.1, w: 8.5, h: 2.2,
            fontSize: TYPE.hero, bold: false, color: COLORS.white,
            fontFace: FONTS.heading, valign: 'top',
            lineSpacing: TYPE.hero * 1.15
        });

        // ArXiv ID badge - refined pill shape
        const badgeW = 1.6;
        slide.addShape('roundRect', {
            x: SLIDE.marginX, y: 3.5, w: badgeW, h: 0.32,
            fill: { color: COLORS.gold },
            rectRadius: 0.16
        });
        slide.addText(this.data.arxiv_id, {
            x: SLIDE.marginX, y: 3.5, w: badgeW, h: 0.32,
            fontSize: TYPE.tiny, bold: true, color: COLORS.navy,
            fontFace: FONTS.accent, align: 'center', valign: 'middle'
        });

        // Authors - elegant formatting
        const authors = this.data.authors.slice(0, 3).join('  ·  ') +
            (this.data.authors.length > 3 ? '  · et al.' : '');
        slide.addText(authors, {
            x: SLIDE.marginX, y: 3.95, w: 8, h: 0.32,
            fontSize: TYPE.small, color: COLORS.muted,
            fontFace: FONTS.body
        });

        // Stats bar - refined metrics display
        const stats = [];
        if (this.data.learning_time) stats.push(`◷ ${this.data.learning_time}`);
        if (this.data.concepts) stats.push(`◈ ${this.data.concepts.length} Concepts`);
        if (this.data.total_words) stats.push(`≡ ${Math.round(this.data.total_words / 100) / 10}k words`);

        if (stats.length > 0) {
            slide.addText(stats.join('     '), {
                x: SLIDE.marginX, y: 4.4, w: 6, h: 0.28,
                fontSize: TYPE.small, color: COLORS.steel,
                fontFace: FONTS.body
            });
        }

        // Premium branding - bottom right
        slide.addText('Jotty AI', {
            x: 7.8, y: 5.0, w: 1.8, h: 0.22,
            fontSize: TYPE.tiny, bold: true, color: COLORS.goldDark,
            fontFace: FONTS.accent, align: 'right'
        });
        slide.addText('Research Intelligence', {
            x: 7.8, y: 5.18, w: 1.8, h: 0.18,
            fontSize: TYPE.micro, color: COLORS.steel,
            fontFace: FONTS.body, align: 'right'
        });
    }

    addAgendaSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'AGENDA');

        // Section number watermark
        slide.addText('02', {
            x: 7.5, y: 0.8, w: 2.5, h: 1.2,
            fontSize: 72, color: COLORS.lightGray,
            fontFace: FONTS.heading, align: 'right',
            transparency: 50
        });

        // Agenda items - clean numbered list
        const items = [
            { label: 'Why This Matters', desc: 'The problem and opportunity', color: COLORS.blue },
            { label: 'Key Concepts', desc: 'Core ideas explained simply', color: COLORS.success },
            { label: 'Deep Dives', desc: 'Understanding how it works', color: COLORS.warning },
            { label: `${this.data.bingo_word || 'Key'} Insights`, desc: 'The breakthrough moments', color: COLORS.gold },
            { label: 'Next Steps', desc: 'Where to go from here', color: COLORS.blueDark }
        ];

        items.forEach((item, i) => {
            const y = 1.0 + i * 0.82;
            const isAlt = i % 2 === 1;

            // Subtle alternating background
            if (isAlt) {
                slide.addShape('rect', {
                    x: SLIDE.marginX - 0.1, y: y - 0.08, w: SLIDE.contentWidth + 0.2, h: 0.72,
                    fill: { color: COLORS.offWhite }
                });
            }

            // Colored left accent bar
            slide.addShape('rect', {
                x: SLIDE.marginX - 0.1, y: y - 0.08, w: 0.04, h: 0.72,
                fill: { color: item.color }
            });

            // Number badge only - proper circle
            const circleSize = 0.36;
            slide.addShape('ellipse', {
                x: SLIDE.marginX + 0.08, y: y + 0.09, w: circleSize, h: circleSize,
                fill: { color: item.color }
            });
            slide.addText(`${i + 1}`, {
                x: SLIDE.marginX + 0.08, y: y + 0.09, w: circleSize, h: circleSize,
                fontSize: TYPE.body, bold: true, color: COLORS.white,
                fontFace: FONTS.accent, align: 'center', valign: 'middle'
            });

            // Label - directly after number
            slide.addText(item.label, {
                x: SLIDE.marginX + 0.55, y: y + 0.1, w: 3.5, h: 0.35,
                fontSize: TYPE.body + 1, bold: true, color: COLORS.slate,
                fontFace: FONTS.accent
            });

            // Description - muted
            slide.addText(item.desc, {
                x: 4.3, y: y + 0.14, w: 5.0, h: 0.28,
                fontSize: TYPE.small, color: COLORS.muted,
                fontFace: FONTS.body
            });
        });

        this.addSlideFooter(slide);
    }

    addHookSlide() {
        if (!this.data.hook) return;

        const slide = this.pptx.addSlide({ masterName: 'MASTER_QUOTE' });
        this.slideNumber++;

        // Large quotation mark watermark
        slide.addText('"', {
            x: 0.2, y: 0.3, w: 2, h: 1.8,
            fontSize: 180, color: COLORS.navyMid,
            fontFace: 'Georgia', transparency: 60
        });

        // Section label - refined
        slide.addText('WHY THIS MATTERS', {
            x: SLIDE.marginX, y: 0.65, w: 4, h: 0.28,
            fontSize: TYPE.tiny, bold: true, color: COLORS.gold,
            fontFace: FONTS.accent, charSpacing: 3
        });

        // Decorative corner elements
        slide.addShape('rect', {
            x: 9.0, y: 0.5, w: 0.5, h: 0.012,
            fill: { color: COLORS.goldLight }
        });
        slide.addShape('rect', {
            x: 9.48, y: 0.5, w: 0.012, h: 0.5,
            fill: { color: COLORS.goldLight }
        });

        // Hook content - auto-fit font size based on content length
        const hookWidth = 8.2;
        const hookHeight = 3.2;
        const hookFontSize = TEXT_UTILS.fitFontSize(this.data.hook, hookWidth, hookHeight, TYPE.h2, 14);
        const hookText = this.wrapText(this.data.hook, Math.floor(hookWidth * TEXT_UTILS.charsPerInch(hookFontSize)));

        slide.addText(hookText, {
            x: SLIDE.marginX, y: 1.15, w: hookWidth, h: hookHeight,
            fontSize: hookFontSize, color: COLORS.white,
            fontFace: FONTS.body, valign: 'top',
            lineSpacing: hookFontSize * 1.45
        });

        // Closing quotation mark - positioned based on content
        slide.addText('"', {
            x: 7.8, y: 4.0, w: 1.5, h: 0.8,
            fontSize: 80, color: COLORS.navyMid,
            fontFace: 'Georgia', align: 'right', transparency: 60
        });
    }

    addConceptsOverviewSlide() {
        if (!this.data.concepts || this.data.concepts.length === 0) return;

        const slide = this.pptx.addSlide({ masterName: 'MASTER_DATA' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'KEY CONCEPTS');

        // Section watermark
        slide.addText('◈', {
            x: 8.5, y: 0.75, w: 1.5, h: 1,
            fontSize: 60, color: COLORS.lightGray,
            fontFace: FONTS.body, transparency: 50
        });

        const concepts = this.data.concepts.slice(0, 6);
        const cols = concepts.length <= 3 ? concepts.length : 3;
        const rows = Math.ceil(concepts.length / cols);
        const gutter = 0.18;
        const cardW = (SLIDE.contentWidth - (cols - 1) * gutter) / cols;
        const cardH = rows === 1 ? 3.6 : 1.72;

        concepts.forEach((concept, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = SLIDE.marginX + col * (cardW + gutter);
            const y = 0.95 + row * (cardH + 0.12);

            // Card shadow (subtle)
            slide.addShape('rect', {
                x: x + 0.02, y: y + 0.02, w: cardW, h: cardH,
                fill: { color: COLORS.mediumGray },
                rectRadius: 0.04
            });

            // Card background
            slide.addShape('rect', {
                x, y, w: cardW, h: cardH,
                fill: { color: COLORS.white },
                line: { color: COLORS.lightGray, pt: 0.75 },
                rectRadius: 0.04
            });

            // Difficulty indicator - refined top bar with gradient effect
            const diffColor = concept.difficulty === 'easy' ? COLORS.success :
                              concept.difficulty === 'medium' ? COLORS.gold : COLORS.blue;
            const diffLight = concept.difficulty === 'easy' ? COLORS.successLight :
                              concept.difficulty === 'medium' ? COLORS.goldLight : COLORS.blueLight;

            slide.addShape('rect', {
                x, y, w: cardW, h: 0.06,
                fill: { color: diffColor }
            });
            slide.addShape('rect', {
                x, y: y + 0.06, w: cardW, h: 0.02,
                fill: { color: diffLight }
            });

            // Concept number - top left
            slide.addText(`${i + 1}`, {
                x: x + 0.1, y: y + 0.12, w: 0.22, h: 0.22,
                fontSize: TYPE.small, bold: true, color: COLORS.muted,
                fontFace: FONTS.accent
            });

            // Concept name - full name with word wrap (multi-line allowed)
            slide.addText(concept.name, {
                x: x + 0.32, y: y + 0.12, w: cardW - 0.42, h: 0.45,
                fontSize: TYPE.body - 0.5, bold: true, color: COLORS.navy,
                fontFace: FONTS.accent, valign: 'top',
                lineSpacing: TYPE.body * 1.1
            });

            // Difficulty badge - below title, left aligned
            const diffLabel = concept.difficulty === 'easy' ? 'Foundational' :
                              concept.difficulty === 'medium' ? 'Intermediate' : 'Advanced';
            slide.addText(diffLabel, {
                x: x + 0.1, y: y + 0.58, w: 1.0, h: 0.16,
                fontSize: TYPE.micro, bold: true, color: diffColor,
                fontFace: FONTS.body
            });

            // Description
            const descLines = rows === 1 ? 220 : 90;
            slide.addText(this.truncate(concept.description || '', descLines), {
                x: x + 0.1, y: y + 0.78, w: cardW - 0.2, h: cardH - 0.9,
                fontSize: TYPE.small - 0.5, color: COLORS.steel,
                fontFace: FONTS.body, valign: 'top',
                lineSpacing: TYPE.small * 1.3
            });
        });

        this.addSlideFooter(slide);
    }

    addConceptDeepDive(concept, index) {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, `CONCEPT ${String(index + 1).padStart(2, '0')}`);

        // Large watermark number
        slide.addText(String(index + 1).padStart(2, '0'), {
            x: 7.2, y: 0.7, w: 2.8, h: 1.5,
            fontSize: 85, color: COLORS.lightGray,
            fontFace: FONTS.heading, align: 'right',
            transparency: 60
        });

        // Concept name - prominent
        slide.addText(concept.name, {
            x: SLIDE.marginX, y: 0.82, w: 7, h: 0.55,
            fontSize: TYPE.h2, bold: true, color: COLORS.navy,
            fontFace: FONTS.accent
        });

        // Difficulty badge - refined pill
        const diffLabel = concept.difficulty === 'easy' ? 'FOUNDATIONAL' :
                         concept.difficulty === 'medium' ? 'INTERMEDIATE' : 'ADVANCED';
        const diffColor = concept.difficulty === 'easy' ? COLORS.success :
                         concept.difficulty === 'medium' ? COLORS.gold : COLORS.blue;

        slide.addShape('roundRect', {
            x: SLIDE.marginX, y: 1.4, w: 1.15, h: 0.24,
            fill: { color: diffColor },
            rectRadius: 0.12
        });
        slide.addText(diffLabel, {
            x: SLIDE.marginX, y: 1.4, w: 1.15, h: 0.24,
            fontSize: TYPE.micro, bold: true, color: COLORS.white,
            fontFace: FONTS.accent, align: 'center', valign: 'middle'
        });

        // Description with left accent bar
        slide.addShape('rect', {
            x: SLIDE.marginX, y: 1.78, w: 0.035, h: 1.25,
            fill: { color: diffColor }
        });
        slide.addText(this.truncate(concept.description || '', 550), {
            x: SLIDE.marginX + 0.18, y: 1.78, w: SLIDE.contentWidth - 0.18, h: 1.3,
            fontSize: TYPE.body, color: COLORS.slate,
            fontFace: FONTS.body, valign: 'top',
            lineSpacing: TYPE.body * 1.55
        });

        // "Why It Matters" box - premium callout design
        if (concept.why_it_matters) {
            const boxY = 3.18;
            const boxH = 1.85;

            // Box background with subtle gradient effect
            slide.addShape('rect', {
                x: SLIDE.marginX, y: boxY, w: SLIDE.contentWidth, h: boxH,
                fill: { color: 'fffdf7' },
                line: { color: COLORS.goldLight, pt: 1 },
                rectRadius: 0.06
            });

            // Gold left accent
            slide.addShape('rect', {
                x: SLIDE.marginX, y: boxY, w: 0.06, h: boxH,
                fill: { color: COLORS.gold }
            });

            // Icon and label
            slide.addText('▸', {
                x: SLIDE.marginX + 0.2, y: boxY + 0.12, w: 0.25, h: 0.25,
                fontSize: TYPE.h3, bold: true, color: COLORS.gold,
                fontFace: FONTS.body
            });
            slide.addText('WHY IT MATTERS', {
                x: SLIDE.marginX + 0.45, y: boxY + 0.15, w: 2.5, h: 0.25,
                fontSize: TYPE.tiny, bold: true, color: COLORS.gold,
                fontFace: FONTS.accent, charSpacing: 1.5
            });

            slide.addText(this.truncate(concept.why_it_matters, 420), {
                x: SLIDE.marginX + 0.25, y: boxY + 0.5, w: SLIDE.contentWidth - 0.45, h: boxH - 0.6,
                fontSize: TYPE.body - 0.5, color: COLORS.slate,
                fontFace: FONTS.body, valign: 'top',
                lineSpacing: TYPE.body * 1.45
            });
        }

        this.addSlideFooter(slide);
    }

    addSectionSlide(section) {
        const isBingo = section.has_bingo_moment;
        const masterName = isBingo ? 'MASTER_INSIGHT' : 'MASTER_CONTENT';
        const slide = this.pptx.addSlide({ masterName });
        this.slideNumber++;

        this.addSlideHeader(slide, section.title.toUpperCase().slice(0, 45), COLORS.white);

        // Bingo badge if applicable - refined design
        if (isBingo) {
            slide.addShape('roundRect', {
                x: 7.8, y: 0.12, w: 1.65, h: 0.28,
                fill: { color: COLORS.gold },
                rectRadius: 0.14
            });
            slide.addText(`★ ${(this.data.bingo_word || 'KEY').toUpperCase()}`, {
                x: 7.8, y: 0.12, w: 1.65, h: 0.28,
                fontSize: TYPE.micro, bold: true, color: COLORS.navy,
                fontFace: FONTS.accent, align: 'center', valign: 'middle'
            });
        }

        // Content area
        const hasCode = section.code_example && section.code_example.trim();
        const contentH = hasCode ? 1.95 : 3.9;
        const contentX = isBingo ? SLIDE.marginX + 0.15 : SLIDE.marginX;

        slide.addText(this.truncate(section.content || '', 850), {
            x: contentX, y: 0.92, w: SLIDE.contentWidth - 0.15, h: contentH,
            fontSize: TYPE.body, color: COLORS.slate,
            fontFace: FONTS.body, valign: 'top',
            lineSpacing: TYPE.body * 1.55,
            paraSpaceAfter: 10
        });

        // Code block if present - GitHub Light style (readable without syntax highlighting)
        if (hasCode) {
            const codeY = 3.0;
            const codeH = 2.05;  // Slightly reduced to ensure fits above footer

            // GitHub Light theme - dark text on light background (better readability without syntax highlighting)
            const codeTheme = {
                bg: 'f6f8fa',         // GitHub light gray background
                fg: '24292e',         // GitHub dark text
                accent: '0366d6',     // GitHub blue accent
                border: 'e1e4e8'      // Light border
            };

            // Code container - light background
            slide.addShape('rect', {
                x: SLIDE.marginX, y: codeY, w: SLIDE.contentWidth, h: codeH,
                fill: { color: codeTheme.bg },
                line: { color: codeTheme.border, pt: 1 },
                rectRadius: 0.05
            });

            // Blue accent bar on left
            slide.addShape('rect', {
                x: SLIDE.marginX, y: codeY, w: 0.05, h: codeH,
                fill: { color: codeTheme.accent }
            });

            // Code content - dark text on light background
            const formattedCode = this.formatCode(section.code_example, 10);  // Reduced lines to prevent overflow
            slide.addText(formattedCode, {
                x: SLIDE.marginX + 0.18, y: codeY + 0.1, w: SLIDE.contentWidth - 0.35, h: codeH - 0.15,
                fontSize: TYPE.small - 1, color: codeTheme.fg,
                fontFace: FONTS.code, valign: 'top',
                lineSpacing: TYPE.small * 1.35
            });
        }

        this.addSlideFooter(slide);
    }

    addInsightsSlide() {
        if (!this.data.key_insights || this.data.key_insights.length === 0) return;

        const slide = this.pptx.addSlide({ masterName: 'MASTER_INSIGHT' });
        this.slideNumber++;
        this.addSlideHeader(slide, `${(this.data.bingo_word || 'KEY').toUpperCase()} INSIGHTS`, COLORS.white);

        // Decorative star watermark
        slide.addText('★', {
            x: 8.0, y: 0.7, w: 2, h: 1.5,
            fontSize: 100, color: COLORS.successLight,
            fontFace: FONTS.body, transparency: 85
        });

        const insights = this.data.key_insights.slice(0, 5);

        insights.forEach((insight, i) => {
            const y = 0.95 + i * 0.82;
            const isAlt = i % 2 === 1;

            // Subtle alternating background
            if (isAlt) {
                slide.addShape('rect', {
                    x: SLIDE.marginX + 0.1, y: y - 0.05, w: SLIDE.contentWidth - 0.2, h: 0.72,
                    fill: { color: COLORS.offWhite },
                    rectRadius: 0.04
                });
            }

            // Number badge - elegant design
            slide.addShape('ellipse', {
                x: SLIDE.marginX + 0.15, y: y + 0.08, w: 0.42, h: 0.42,
                fill: { color: COLORS.gold }
            });
            // Inner circle for depth
            slide.addShape('ellipse', {
                x: SLIDE.marginX + 0.19, y: y + 0.12, w: 0.34, h: 0.34,
                line: { color: COLORS.goldDark, pt: 1.5 }
            });
            slide.addText(`${i + 1}`, {
                x: SLIDE.marginX + 0.15, y: y + 0.08, w: 0.42, h: 0.42,
                fontSize: TYPE.body + 1, bold: true, color: COLORS.navy,
                fontFace: FONTS.accent, align: 'center', valign: 'middle'
            });

            // Insight text
            slide.addText(this.truncate(insight, 200), {
                x: SLIDE.marginX + 0.75, y: y + 0.08, w: SLIDE.contentWidth - 0.9, h: 0.68,
                fontSize: TYPE.body, color: COLORS.slate,
                fontFace: FONTS.body, valign: 'top',
                lineSpacing: TYPE.body * 1.45
            });
        });

        this.addSlideFooter(slide);
    }

    addSummarySlide() {
        if (!this.data.summary) return;

        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'EXECUTIVE SUMMARY');

        // Decorative element
        slide.addText('≡', {
            x: 8.5, y: 0.75, w: 1.5, h: 1,
            fontSize: 55, color: COLORS.lightGray,
            fontFace: FONTS.body, transparency: 50
        });

        // Summary box with refined styling
        slide.addShape('rect', {
            x: SLIDE.marginX, y: 0.9, w: 0.04, h: 4.0,
            fill: { color: COLORS.blue }
        });

        // Large summary text with premium typography
        slide.addText(this.truncate(this.data.summary, 950), {
            x: SLIDE.marginX + 0.2, y: 0.95, w: SLIDE.contentWidth - 0.2, h: 3.95,
            fontSize: TYPE.body + 2, color: COLORS.slate,
            fontFace: FONTS.body, valign: 'top',
            lineSpacing: (TYPE.body + 2) * 1.65,
            paraSpaceAfter: 14
        });

        this.addSlideFooter(slide);
    }

    addNextStepsSlide() {
        if (!this.data.next_steps || this.data.next_steps.length === 0) return;

        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'NEXT STEPS');

        const steps = this.data.next_steps.slice(0, 5);

        steps.forEach((step, i) => {
            const y = 0.95 + i * 0.78;
            const isAlt = i % 2 === 1;

            // Alternating background
            if (isAlt) {
                slide.addShape('rect', {
                    x: SLIDE.marginX - 0.05, y: y - 0.05, w: SLIDE.contentWidth + 0.1, h: 0.7,
                    fill: { color: COLORS.offWhite },
                    rectRadius: 0.04
                });
            }

            // Step number in circle (equal width and height for perfect circle)
            const circleSize = 0.35;
            slide.addShape('ellipse', {
                x: SLIDE.marginX, y: y + 0.1, w: circleSize, h: circleSize,
                fill: { color: COLORS.blue }
            });
            slide.addText(`${i + 1}`, {
                x: SLIDE.marginX, y: y + 0.1, w: circleSize, h: circleSize,
                fontSize: TYPE.body, bold: true, color: COLORS.white,
                fontFace: FONTS.accent, align: 'center', valign: 'middle'
            });

            // Step text - directly after number
            slide.addText(this.truncate(step, 140), {
                x: SLIDE.marginX + 0.5, y: y + 0.12, w: SLIDE.contentWidth - 0.6, h: 0.55,
                fontSize: TYPE.body + 0.5, color: COLORS.slate,
                fontFace: FONTS.body, valign: 'top',
                lineSpacing: TYPE.body * 1.35
            });
        });

        this.addSlideFooter(slide);
    }

    // ========================================================================
    // VISUALIZATION SLIDES - World-Class Visual Storytelling
    // ========================================================================

    /**
     * Architecture Diagram Slide - Shows system/model architecture
     * Uses shape-based rendering for precise control over layout
     */
    addArchitectureDiagramSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'ARCHITECTURE OVERVIEW');

        // Decorative element
        slide.addText('⬡', {
            x: 8.5, y: 0.7, w: 1.5, h: 1,
            fontSize: 50, color: COLORS.lightGray,
            fontFace: FONTS.body, transparency: 60
        });

        const diagram = new DiagramBuilder(slide, SLIDE.marginX, 0.9, SLIDE.contentWidth, 4.2);

        // Create architecture layers based on paper content
        const layers = this.generateArchitectureLayers();
        diagram.drawArchitectureDiagram(layers, {
            boxWidth: 1.3,
            boxHeight: 0.52,
            layerGap: 0.25
        });

        this.addSlideFooter(slide);
    }

    /**
     * Generate architecture layers from LIDA specs or paper concepts
     *
     * LIDA-style: Uses detailed LLM-generated specs when available
     * Fallback: Pattern-based generation from concepts
     */
    generateArchitectureLayers() {
        // PRIORITY 1: Use LIDA-style visualization specs if available
        const vizSpecs = this.data.visualization_specs || {};
        const archSpec = vizSpecs.architecture;

        if (archSpec && archSpec.nodes && archSpec.nodes.length > 0) {
            // Use LLM-generated architecture spec
            console.log(`📐 Using LIDA architecture spec: ${archSpec.nodes.length} nodes`);

            // Group nodes by row/layer
            const nodesByRow = {};
            archSpec.nodes.forEach(node => {
                const row = node.row || 0;
                if (!nodesByRow[row]) nodesByRow[row] = [];
                nodesByRow[row].push({
                    label: this.truncate(node.label || node.id, 22),
                    subtitle: node.sublabel || ''
                });
            });

            // Convert to layers array sorted by row
            const rows = Object.keys(nodesByRow).map(Number).sort((a, b) => a - b);
            return rows.map(row => nodesByRow[row]);
        }

        // PRIORITY 2: Fallback to pattern-based generation
        const concepts = this.data.concepts || [];
        const conceptNames = concepts.map(c => c.name.toLowerCase());

        // Check for common architecture patterns
        const hasEncoder = conceptNames.some(n => n.includes('encoder'));
        const hasDecoder = conceptNames.some(n => n.includes('decoder'));
        const hasAttention = conceptNames.some(n => n.includes('attention'));
        const hasEmbedding = conceptNames.some(n => n.includes('embed') || n.includes('position'));

        // Build layers based on detected patterns
        if (hasEncoder && hasDecoder) {
            // Transformer-like architecture
            return [
                [{ label: 'Input', subtitle: 'Tokens' }, { label: 'Input', subtitle: 'Tokens' }],
                [{ label: 'Embedding', subtitle: '+ Positional' }, { label: 'Embedding', subtitle: '+ Positional' }],
                [{ label: 'Encoder', subtitle: 'Self-Attention' }, { label: 'Decoder', subtitle: 'Cross-Attention' }],
                [{ label: 'Output', subtitle: 'Predictions' }]
            ];
        } else if (hasAttention) {
            // Attention-based model
            return [
                [{ label: 'Input Sequence', subtitle: 'Embeddings' }],
                [{ label: 'Query (Q)', subtitle: '' }, { label: 'Key (K)', subtitle: '' }, { label: 'Value (V)', subtitle: '' }],
                [{ label: 'Attention Scores', subtitle: 'softmax(QK^T/√d)' }],
                [{ label: 'Output', subtitle: 'Weighted Values' }]
            ];
        } else {
            // Generic neural network
            const layerNames = concepts.slice(0, 4).map(c => ({
                label: this.truncate(c.name, 20),
                subtitle: c.difficulty || ''
            }));
            return layerNames.length > 0 ? layerNames.map(l => [l]) : [
                [{ label: 'Input Layer', subtitle: '' }],
                [{ label: 'Hidden Layer', subtitle: '' }],
                [{ label: 'Output Layer', subtitle: '' }]
            ];
        }
    }

    /**
     * Concept Relationship Slide - Shows how concepts connect
     * Uses shape-based rendering with cardinal layout (no overlapping lines)
     */
    addConceptMapSlide() {
        if (!this.data.concepts || this.data.concepts.length < 3) return;

        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'CONCEPT RELATIONSHIPS');

        const diagram = new DiagramBuilder(slide, SLIDE.marginX, 0.85, SLIDE.contentWidth, 4.2);

        // Use first concept as center, others as related
        const concepts = this.data.concepts.slice(0, 5);
        const centerConcept = concepts[0].name;
        const relatedConcepts = concepts.slice(1).map(c => ({
            label: this.truncate(c.name, 28)
        }));

        diagram.drawConceptMap(this.truncate(centerConcept, 32), relatedConcepts, {
            radius: 1.85,
            centerSize: 2.0,
            nodeSize: 1.4
        });

        this.addSlideFooter(slide);
    }

    /**
     * Flow Diagram Slide - Shows process/data flow
     * Uses Mermaid-generated image if available, otherwise draws shapes
     */
    addFlowDiagramSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'HOW IT WORKS');

        // Check for Mermaid-generated image
        const vizSpecs = this.data.visualization_specs || {};
        const diagramImages = vizSpecs.diagram_images || {};
        const flowImage = diagramImages.flow;

        if (flowImage && fs.existsSync(flowImage)) {
            // Use pre-generated Mermaid flow diagram
            // Center horizontally, maintain aspect ratio (no stretching)
            console.log(`🖼️ Using Mermaid flow diagram: ${flowImage}`);

            // Flow diagrams are typically wide and short (horizontal flowchart)
            // Use contain sizing to maintain aspect ratio
            const maxWidth = SLIDE.contentWidth;
            const maxHeight = 2.5;  // Shorter height for horizontal flow
            const centerY = 1.5 + (3.5 - maxHeight) / 2;  // Center vertically in available space

            slide.addImage({
                path: flowImage,
                x: SLIDE.marginX,
                y: centerY,
                w: maxWidth,
                h: maxHeight,
                sizing: { type: 'contain', w: maxWidth, h: maxHeight }
            });
        } else {
            // Fallback: Draw shapes
            // Decorative element
            slide.addText('→', {
                x: 8.5, y: 0.7, w: 1.5, h: 1,
                fontSize: 45, color: COLORS.lightGray,
                fontFace: FONTS.body, transparency: 60
            });

            const diagram = new DiagramBuilder(slide, SLIDE.marginX, 1.2, SLIDE.contentWidth, 3.5);

            // Generate flow nodes from sections or concepts
            const flowNodes = this.generateFlowNodes();
            diagram.drawFlowDiagram(flowNodes, {
                nodeWidth: 1.8,
                nodeHeight: 0.85,
                gap: 0.35
            });
        }

        this.addSlideFooter(slide);
    }

    /**
     * Generate flow nodes from paper content
     */
    generateFlowNodes() {
        // Try to extract process steps from sections
        const sections = this.data.sections || [];
        const concepts = this.data.concepts || [];

        // Use first 4-5 key items for flow
        if (sections.length >= 3) {
            return sections.slice(0, 4).map((s, i) => ({
                label: this.truncate(s.title, 28),
                subtitle: `Step ${i + 1}`,
                arrowLabel: i < 3 ? '' : null
            }));
        } else if (concepts.length >= 3) {
            return concepts.slice(0, 4).map((c, i) => ({
                label: this.truncate(c.name, 28),
                subtitle: c.difficulty || '',
                arrowLabel: i < 3 ? '' : null
            }));
        }

        // Default flow
        return [
            { label: 'Input', subtitle: 'Data', arrowLabel: '' },
            { label: 'Process', subtitle: 'Transform', arrowLabel: '' },
            { label: 'Learn', subtitle: 'Train', arrowLabel: '' },
            { label: 'Output', subtitle: 'Results' }
        ];
    }

    /**
     * Metrics/Stats Slide - Shows key numbers
     * Uses shape-based rendering for clean metric cards
     */
    addMetricsSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_DATA' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'KEY METRICS');

        const diagram = new DiagramBuilder(slide, SLIDE.marginX + 0.5, 1.1, SLIDE.contentWidth - 1, 1.5);

        // Generate metrics from paper data
        const metrics = this.generateMetrics();
        diagram.drawMetricCards(metrics);

        // Add context below metrics
        if (this.data.summary) {
            slide.addText(this.truncate(this.data.summary, 300), {
                x: SLIDE.marginX, y: 3.0, w: SLIDE.contentWidth, h: 2.0,
                fontSize: TYPE.body - 1, color: COLORS.steel,
                fontFace: FONTS.body, valign: 'top',
                lineSpacing: TYPE.body * 1.5
            });
        }

        this.addSlideFooter(slide);
    }

    /**
     * Generate metrics from paper content
     */
    generateMetrics() {
        const metrics = [];

        // Concepts count
        if (this.data.concepts) {
            metrics.push({
                value: String(this.data.concepts.length),
                label: 'Core Concepts',
                subtitle: 'Key ideas covered'
            });
        }

        // Insights count
        if (this.data.key_insights) {
            metrics.push({
                value: String(this.data.key_insights.length),
                label: 'Key Insights',
                subtitle: 'Breakthrough moments'
            });
        }

        // Learning time
        if (this.data.learning_time) {
            const time = this.data.learning_time.replace(/\s*min(utes)?/i, '');
            metrics.push({
                value: time,
                label: 'Minutes',
                subtitle: 'Learning time'
            });
        }

        // Word count
        if (this.data.total_words) {
            const kWords = (this.data.total_words / 1000).toFixed(1);
            metrics.push({
                value: `${kWords}k`,
                label: 'Words',
                subtitle: 'Content depth'
            });
        }

        // Ensure at least 3 metrics
        while (metrics.length < 3) {
            metrics.push({
                value: '★',
                label: 'Quality',
                subtitle: 'Research-grade'
            });
        }

        return metrics.slice(0, 4);
    }

    /**
     * Comparison Slide - Before/After or Traditional vs New
     * Uses shape-based rendering for precise two-column layout
     */
    addComparisonSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'INNOVATION COMPARISON');

        const diagram = new DiagramBuilder(slide, SLIDE.marginX, 0.95, SLIDE.contentWidth, 4.0);

        // Generate comparison based on paper context
        const comparison = this.generateComparison();
        diagram.drawComparison(comparison.left, comparison.right, { vsLabel: 'vs' });

        this.addSlideFooter(slide);
    }

    /**
     * Generate comparison content from LIDA specs or patterns
     *
     * LIDA-style: Uses detailed LLM-generated specs with actual paper comparisons
     * Fallback: Pattern-based generation
     */
    generateComparison() {
        // PRIORITY 1: Use LIDA-style visualization specs if available
        const vizSpecs = this.data.visualization_specs || {};
        const compSpec = vizSpecs.comparison;

        if (compSpec && compSpec.left_items && compSpec.left_items.length > 0) {
            console.log(`📊 Using LIDA comparison spec: ${compSpec.left_items.length} vs ${compSpec.right_items.length}`);
            return {
                left: {
                    title: compSpec.left_title || 'Traditional Approach',
                    points: compSpec.left_items.map(item =>
                        typeof item === 'string' ? item : item.point || item.text || String(item)
                    ).slice(0, 5)
                },
                right: {
                    title: compSpec.right_title || 'New Approach',
                    points: compSpec.right_items.map(item =>
                        typeof item === 'string' ? item : item.point || item.text || String(item)
                    ).slice(0, 5)
                }
            };
        }

        // PRIORITY 2: Fallback - Look for comparative language in hook/summary
        const text = (this.data.hook || '') + ' ' + (this.data.summary || '');
        const hasTransformer = text.toLowerCase().includes('transformer');
        const hasAttention = text.toLowerCase().includes('attention');
        const hasRNN = text.toLowerCase().includes('rnn') || text.toLowerCase().includes('recurrent');

        if (hasTransformer || hasAttention) {
            return {
                left: {
                    title: 'Traditional Approach',
                    points: [
                        'Sequential processing',
                        'Limited parallelization',
                        'Long-range dependency issues',
                        'Slower training times'
                    ]
                },
                right: {
                    title: 'New Approach',
                    points: [
                        'Parallel processing',
                        'Full parallelization',
                        'Direct long-range connections',
                        'Faster training (10x+)'
                    ]
                }
            };
        }

        // Generic comparison
        return {
            left: {
                title: 'Before',
                points: [
                    'Previous limitations',
                    'Complex architectures',
                    'Slower convergence',
                    'Higher compute costs'
                ]
            },
            right: {
                title: 'After',
                points: [
                    'Improved capabilities',
                    'Elegant simplicity',
                    'Faster convergence',
                    'More efficient'
                ]
            }
        };
    }

    /**
     * Timeline Slide - Evolution/History
     */
    addTimelineSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_CONTENT' });
        this.slideNumber++;
        this.addSlideHeader(slide, 'EVOLUTION & IMPACT');

        const diagram = new DiagramBuilder(slide, SLIDE.marginX, 1.3, SLIDE.contentWidth, 3.5);

        // Generate timeline events
        const events = this.generateTimelineEvents();
        diagram.drawTimeline(events);

        this.addSlideFooter(slide);
    }

    /**
     * Generate timeline events
     */
    generateTimelineEvents() {
        // Check for year references in the paper
        const arxivId = this.data.arxiv_id || '';
        const year = arxivId.substring(0, 2);
        const fullYear = parseInt(year) > 50 ? `19${year}` : `20${year}`;

        return [
            { label: 'Problem', subtitle: 'Identified' },
            { label: 'Research', subtitle: 'Development' },
            { label: fullYear, subtitle: 'Published' },
            { label: 'Adoption', subtitle: 'Industry use' },
            { label: 'Impact', subtitle: 'Transformative' }
        ];
    }

    addClosingSlide() {
        const slide = this.pptx.addSlide({ masterName: 'MASTER_TITLE' });
        this.slideNumber++;

        // Large decorative element
        slide.addText('◈', {
            x: 4, y: 0.6, w: 2, h: 1.2,
            fontSize: 65, color: COLORS.navyMid,
            fontFace: FONTS.body, align: 'center', transparency: 60
        });

        // Thank you - cinematic typography
        slide.addText('Thank You', {
            x: 0.5, y: 1.4, w: 9, h: 0.9,
            fontSize: 52, bold: false, color: COLORS.white,
            fontFace: FONTS.heading, align: 'center'
        });

        // Elegant divider line
        slide.addShape('rect', {
            x: 4, y: 2.4, w: 2, h: 0.015,
            fill: { color: COLORS.gold }
        });

        // Paper reference - refined
        slide.addText(this.truncate(this.data.paper_title, 65), {
            x: 1, y: 2.65, w: 8, h: 0.5,
            fontSize: TYPE.body + 1, color: COLORS.muted,
            fontFace: FONTS.body, align: 'center'
        });

        // ArXiv badge
        slide.addShape('roundRect', {
            x: 4.2, y: 3.25, w: 1.6, h: 0.3,
            fill: { color: COLORS.gold },
            rectRadius: 0.15
        });
        slide.addText(this.data.arxiv_id, {
            x: 4.2, y: 3.25, w: 1.6, h: 0.3,
            fontSize: TYPE.small, bold: true, color: COLORS.navy,
            fontFace: FONTS.accent, align: 'center', valign: 'middle'
        });

        // Stats summary - premium design
        const stats = [];
        if (this.data.concepts) stats.push(`${this.data.concepts.length} Concepts`);
        if (this.data.key_insights) stats.push(`${this.data.key_insights.length} Insights`);
        stats.push(`${this.slideNumber} Slides`);

        slide.addText(stats.join('   ·   '), {
            x: 0.5, y: 3.85, w: 9, h: 0.28,
            fontSize: TYPE.small, color: COLORS.steel,
            fontFace: FONTS.body, align: 'center'
        });

        // Premium branding footer
        slide.addShape('rect', {
            x: 3.5, y: 4.6, w: 3, h: 0.01,
            fill: { color: COLORS.goldDark }
        });
        slide.addText('Jotty AI', {
            x: 0.5, y: 4.75, w: 9, h: 0.25,
            fontSize: TYPE.small, bold: true, color: COLORS.gold,
            fontFace: FONTS.accent, align: 'center'
        });
        slide.addText('Research Intelligence Platform', {
            x: 0.5, y: 4.98, w: 9, h: 0.2,
            fontSize: TYPE.micro, color: COLORS.steel,
            fontFace: FONTS.body, align: 'center'
        });
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    addSlideHeader(slide, text, color = COLORS.white) {
        slide.addText(text, {
            x: SLIDE.marginX, y: 0.13, w: 7, h: 0.32,
            fontSize: TYPE.small + 0.5, bold: true, color: color,
            fontFace: FONTS.accent, charSpacing: 1.5
        });
    }

    addSlideFooter(slide) {
        const footerY = SLIDE.footerY + 0.06;

        // Jotty branding - subtle
        slide.addText('Jotty AI', {
            x: SLIDE.marginX, y: footerY, w: 1, h: 0.18,
            fontSize: TYPE.micro, bold: true, color: COLORS.steel,
            fontFace: FONTS.accent
        });

        // ArXiv reference - center
        slide.addText(this.data.arxiv_id, {
            x: 4, y: footerY, w: 2, h: 0.18,
            fontSize: TYPE.micro, color: COLORS.muted,
            fontFace: FONTS.body, align: 'center'
        });

        // Page number - proper circle (equal width and height)
        const circleSize = 0.22;
        slide.addShape('ellipse', {
            x: 9.25, y: footerY - 0.01, w: circleSize, h: circleSize,
            fill: { color: COLORS.navy }
        });
        slide.addText(`${this.slideNumber}`, {
            x: 9.25, y: footerY - 0.01, w: circleSize, h: circleSize,
            fontSize: TYPE.micro, bold: true, color: COLORS.white,
            fontFace: FONTS.accent, align: 'center', valign: 'middle'
        });
    }

    truncate(text, maxLen) {
        if (!text) return '';
        text = String(text).replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
        if (text.length <= maxLen) return text;
        return text.substring(0, maxLen - 3).trim() + '...';
    }

    // Format code - preserves line breaks and structure
    formatCode(code, maxLines = 15) {
        if (!code) return '';
        // Clean up but preserve line breaks
        let lines = String(code).split('\n');
        // Limit number of lines
        if (lines.length > maxLines) {
            lines = lines.slice(0, maxLines);
            lines.push('...');
        }
        // Trim trailing whitespace from each line
        lines = lines.map(line => line.trimEnd());
        return lines.join('\n');
    }

    wrapText(text, lineLen) {
        if (!text) return '';
        text = String(text).replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';

        for (const word of words) {
            if ((currentLine + ' ' + word).trim().length <= lineLen) {
                currentLine = (currentLine + ' ' + word).trim();
            } else {
                if (currentLine) lines.push(currentLine);
                currentLine = word;
            }
        }
        if (currentLine) lines.push(currentLine);

        return lines.join('\n');
    }

    // ========================================================================
    // GENERATE
    // ========================================================================

    async generate(outputPath) {
        // Get diagram decisions (intelligent selection, no force-fit)
        const diagramDecisions = this.data.diagram_decisions || {};
        const shouldInclude = (type) => {
            // If no decisions provided, use smart defaults
            if (!Object.keys(diagramDecisions).length) {
                return this._analyzeNeedFor(type);
            }
            return diagramDecisions[type]?.should_include === true;
        };

        // Track which diagrams were included (for feedback)
        this.diagramsIncluded = [];

        // ===== OPENING SEQUENCE =====
        this.addTitleSlide();
        this.addAgendaSlide();
        this.addHookSlide();

        // ===== VISUALIZATION: Architecture Overview =====
        // Only if paper describes an architecture/system
        if (shouldInclude('architecture')) {
            this.addArchitectureDiagramSlide();
            this.diagramsIncluded.push('architecture');
        }

        // ===== CONCEPTS SECTION =====
        this.addConceptsOverviewSlide();

        // Concept deep dives (max 3 to make room for visuals)
        if (this.data.concepts) {
            this.data.concepts.slice(0, 3).forEach((c, i) => this.addConceptDeepDive(c, i));
        }

        // ===== VISUALIZATION: Concept Relationships =====
        // Only if we have 4+ interrelated concepts
        if (shouldInclude('concept_map')) {
            this.addConceptMapSlide();
            this.diagramsIncluded.push('concept_map');
        }

        // ===== CONTENT SECTIONS =====
        if (this.data.sections) {
            const bingoSections = this.data.sections.filter(s => s.has_bingo_moment).slice(0, 3);
            const otherSections = this.data.sections.filter(s => !s.has_bingo_moment && s.level <= 2).slice(0, 3);
            const allSections = [...bingoSections, ...otherSections].slice(0, 4);
            allSections.forEach(s => this.addSectionSlide(s));
        }

        // ===== VISUALIZATION: How It Works Flow =====
        // Only if content describes a sequential process
        if (shouldInclude('flow')) {
            this.addFlowDiagramSlide();
            this.diagramsIncluded.push('flow');
        }

        // ===== VISUALIZATION: Comparison (Before/After) =====
        // Only if paper compares approaches
        if (shouldInclude('comparison')) {
            this.addComparisonSlide();
            this.diagramsIncluded.push('comparison');
        }

        // ===== INSIGHTS & SUMMARY =====
        this.addInsightsSlide();

        // ===== VISUALIZATION: Key Metrics =====
        // Only if we have meaningful statistics
        if (shouldInclude('metrics')) {
            this.addMetricsSlide();
            this.diagramsIncluded.push('metrics');
        }

        // ===== VISUALIZATION: Timeline & Impact =====
        // STRICT: Only if we have actual dates/historical data
        if (shouldInclude('timeline')) {
            this.addTimelineSlide();
            this.diagramsIncluded.push('timeline');
        }

        // ===== CLOSING SEQUENCE =====
        this.addNextStepsSlide();
        this.addClosingSlide();

        // Save
        await this.pptx.writeFile({ fileName: outputPath });
        return outputPath;
    }

    /**
     * Analyze if a diagram type is needed (fallback when no decisions provided)
     */
    _analyzeNeedFor(type) {
        const text = ((this.data.hook || '') + ' ' + (this.data.summary || '')).toLowerCase();
        const concepts = this.data.concepts || [];
        const sections = this.data.sections || [];

        switch (type) {
            case 'architecture':
                // Need architecture keywords or architectural concepts
                const archKeywords = ['architecture', 'encoder', 'decoder', 'layer', 'model', 'network'];
                return archKeywords.some(kw => text.includes(kw)) ||
                       concepts.some(c => archKeywords.some(kw => (c.name || '').toLowerCase().includes(kw)));

            case 'concept_map':
                // Need 4+ concepts to make a meaningful map
                return concepts.length >= 4;

            case 'flow':
                // Need sequential sections that describe a process
                const flowKeywords = ['step', 'process', 'pipeline', 'how', 'works'];
                const hasSequential = sections.length >= 3;
                const hasFlowLanguage = flowKeywords.some(kw => text.includes(kw));
                return hasSequential && hasFlowLanguage;

            case 'comparison':
                // Need comparative language
                const compareKeywords = ['compared to', 'versus', 'vs', 'traditional', 'previous', 'better than', 'faster'];
                return compareKeywords.some(kw => text.includes(kw));

            case 'metrics':
                // Need quantitative data
                return concepts.length >= 3 || this.data.key_insights?.length >= 3;

            case 'timeline':
                // STRICT: Need actual years/dates
                const yearPattern = /\b(19|20)\d{2}\b/g;
                const years = text.match(yearPattern) || [];
                const uniqueYears = [...new Set(years)];
                return uniqueYears.length >= 2;

            default:
                return false;
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
    const args = process.argv.slice(2);

    if (args.length < 2) {
        console.error('Usage: node generate_pptx.js <json_file> <output_path>');
        process.exit(1);
    }

    const jsonFile = args[0];
    const outputPath = args[1];

    try {
        const jsonData = fs.readFileSync(jsonFile, 'utf8');
        const data = JSON.parse(jsonData);

        const generator = new WorldClassPresentation(data);
        await generator.generate(outputPath);

        console.log(`SUCCESS:${outputPath}`);
    } catch (error) {
        console.error(`ERROR:${error.message}`);
        process.exit(1);
    }
}

main();
