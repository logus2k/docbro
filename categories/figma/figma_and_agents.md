# Figma and Agents

Standardizing **layout structures** (e.g., component hierarchies, grids, spacing systems) for use with LLMs requires a **complementary approach** to DTCG tokens. Since DTCG focuses on **token values** (colors, spacing, etc.) but not structural layouts, you need a way to **define, export, and communicate layout rules** in a machine-readable format. Here’s how to do it:

---

## **1. Define Layout Rules in Your Design System**
To standardize layouts, break them down into **reusable, tokenizable components** and **structural rules**. For example:

### **A. Layout Tokens (DTCG-Compatible)**
Define tokens for layout-related properties (e.g., container widths, grid gaps, breakpoints) in your DTCG file:
```json
{
  "layout": {
    "$description": "Tokens for layout-related properties",
    "container": {
      "max-width": { "$type": "dimension", "$value": { "value": 1200, "unit": "px" } },
      "padding": { "$type": "dimension", "$value": { "value": 16, "unit": "px" } }
    },
    "grid": {
      "gap": { "$type": "dimension", "$value": { "value": 16, "unit": "px" } },
      "columns": { "$type": "number", "$value": 12 }
    },
    "breakpoints": {
      "sm": { "$type": "dimension", "$value": { "value": 640, "unit": "px" } },
      "md": { "$type": "dimension", "$value": { "value": 768, "unit": "px" } },
      "lg": { "$type": "dimension", "$value": { "value": 1024, "unit": "px" } }
    }
  }
}
```

### **B. Layout Templates (Structural Rules)**
Define **reusable layout templates** (e.g., "Sidebar + Main Content", "Grid Card Layout") as **JSON schemas** or **markdown descriptions**. For example:
```json
{
  "templates": {
    "sidebar-layout": {
      "$description": "A layout with a sidebar on the left and main content on the right",
      "structure": {
        "container": {
          "type": "flex",
          "direction": "row",
          "children": [
            {
              "type": "sidebar",
              "width": "{layout.sidebar.width}",
              "background": "{color.background.secondary}"
            },
            {
              "type": "main",
              "flex": 1,
              "padding": "{layout.container.padding}"
            }
          ]
        }
      }
    }
  }
}
```
- This describes the **structure** (flexbox, children, etc.) while referencing DTCG tokens for values.

---

---

## **2. Export Layout Rules for LLM Use**
To make layout rules machine-readable for LLMs, use one of these approaches:

---

### **Option 1: JSON Schema for Layouts**
Create a **JSON schema** that defines:
- **Component types** (e.g., `sidebar`, `card`, `header`).
- **Layout properties** (e.g., `flex`, `grid`, `position`).
- **Token references** (e.g., `{layout.grid.gap}`).

Example:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Layout Template",
  "type": "object",
  "properties": {
    "type": { "type": "string", "enum": ["flex", "grid", "stack"] },
    "direction": { "type": "string", "enum": ["row", "column"] },
    "gap": { "$ref": "#/definitions/tokenReference" },
    "children": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": { "type": "string" },
          "width": { "$ref": "#/definitions/tokenReference" },
          "background": { "$ref": "#/definitions/tokenReference" }
        }
      }
    }
  },
  "definitions": {
    "tokenReference": {
      "type": "string",
      "pattern": "^\\{.*\\}$"  // Matches "{layout.grid.gap}"
    }
  }
}
```
- **Pros**: Structured, validates against your design system.
- **Cons**: Requires schema design and validation.

---

### **Option 2: Markdown or YAML Descriptions**
Define layouts in a **human-readable format** (e.g., Markdown or YAML) that LLMs can parse. For example:
```yaml
# Sidebar Layout
description: A responsive layout with a sidebar and main content area.
structure:
  container:
    type: flex
    direction: row
    gap: "{layout.grid.gap}"
    children:
      - type: sidebar
        width: "{layout.sidebar.width}"
        background: "{color.background.secondary}"
      - type: main
        flex: 1
        padding: "{layout.container.padding}"
```
- **Pros**: Easy to write and maintain.
- **Cons**: Less structured, may require LLM fine-tuning for parsing.

---

### **Option 3: Figma API + DTCG Tokens**
Use the **Figma REST API** to extract:
1. **Node hierarchy** (e.g., frames, components).
2. **Layout properties** (e.g., `x`, `y`, `width`, `height`, `flexbox`).
3. **Token references** (e.g., `fill: {color.primary}`).

Example API response (simplified):
```json
{
  "name": "Sidebar Layout",
  "type": "FRAME",
  "children": [
    {
      "name": "Sidebar",
      "type": "RECTANGLE",
      "width": 200,
      "fill": "{color.background.secondary}"
    },
    {
      "name": "Main Content",
      "type": "RECTANGLE",
      "flex": 1,
      "padding": 16
    }
  ],
  "layout": {
    "direction": "row",
    "gap": 16
  }
}
```
- **Pros**: Directly reflects the Figma design.
- **Cons**: Requires API integration and data transformation.

---
---
### **Option 4: Use a Design System Tool**
Tools like **Storybook**, **Zeroheight**, or **Tokens Studio** can help:
- **Storybook**: Define layout components (e.g., `SidebarLayout`) with props for tokens.
- **Zeroheight**: Document layout templates and link them to DTCG tokens.
- **Tokens Studio**: Export DTCG tokens and use plugins to generate layout code.

---
---
## **3. Provide Layout Rules to the LLM**
Once you’ve standardized your layout rules, provide them to the LLM in one of these ways:

### **A. Combined JSON File**
Merge **DTCG tokens** + **layout templates** into a single JSON file:
```json
{
  "tokens": { ... },  // DTCG tokens
  "templates": { ... }  // Layout templates
}
```
Example LLM prompt:
> *"Use the following design system rules to generate a React component for the 'sidebar-layout' template. Replace all token references (e.g., `{layout.sidebar.width}`) with their values from the `tokens` section."*

### **B. Separate Files**
Provide:
1. A **DTCG JSON file** for tokens.
2. A **layout JSON/YAML file** for structure.

Example LLM prompt:
> *"Use the tokens from `dtcg-tokens.json` and the layout from `sidebar-layout.yaml` to generate a React component. Ensure all token references are resolved."*

### **C. API Integration**
If using the Figma API, provide the **raw API response** + **DTCG tokens** to the LLM and ask it to:
1. Parse the node hierarchy.
2. Replace hardcoded values with DTCG tokens.
3. Generate code (e.g., React, CSS).

---
---
## **4. Example Workflow**
1. **Designer** updates a layout in Figma (e.g., moves a sidebar).
2. **Developer/Designer**:
   - Exports **DTCG tokens** (for styling).
   - Exports **layout template** (e.g., as JSON/YAML or via Figma API).
3. **LLM Prompt**:
   > *"Use the following:
   > - DTCG tokens: [JSON]
   > - Layout template: [JSON/YAML]
   > Generate a React component that implements the layout and resolves all token references (e.g., `{layout.sidebar.width}`)."*
4. **LLM** generates the React component with:
   - Correct **structure** (from the layout template).
   - Correct **styling** (from DTCG tokens).

---
---
## **5. Tools to Automate This**
| Tool               | Purpose                                                                 | Link                                  |
|--------------------|-------------------------------------------------------------------------|---------------------------------------|
| **Figma API**      | Fetch layout structure and token references.                          | [Figma API](https://www.figma.com/developers/api) |
| **Tokens Studio**  | Manage DTCG tokens and export them.                                    | [Tokens Studio](https://tokens.studio/) |
| **Locofy**         | Export Figma layouts as React/Vue code.                                | [Locofy](https://www.locofy.ai/)      |
| **Storybook**      | Document layout components and link to DTCG tokens.                   | [Storybook](https://storybook.js.org/) |
| **Style Dictionary** | Transform DTCG tokens into platform-specific code (CSS, JS, etc.). | [Style Dictionary](https://amzn.github.io/style-dictionary/) |

---
---
## **6. Key Benefits**
- **Consistency**: Layouts and tokens are standardized and reusable.
- **Automation**: LLMs can generate code that adheres to your design system.
- **Scalability**: Update tokens or layouts in one place, and regenerate code everywhere.

---
---
## **7. Limitations**
- **LLM Parsing**: LLMs may struggle with complex JSON/YAML. Simplify or use structured prompts.
- **Figma API**: Requires technical setup to fetch and transform data.
- **Maintenance**: Layout templates need to be kept in sync with design changes.

---
---
### **Recommendation**
Start with **Option 2 (YAML/Markdown descriptions)** for simplicity, then move to **Option 1 (JSON Schema)** or **Option 3 (Figma API)** as your design system matures. Use **Tokens Studio** or **Style Dictionary** to manage DTCG tokens.
