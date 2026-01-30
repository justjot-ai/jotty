"""
Artifacts Builder Skill - Create HTML artifacts with React, TypeScript, Tailwind.

Helps create complex HTML artifacts for Claude.ai using modern frontend
technologies including React, TypeScript, Tailwind CSS, and shadcn/ui.
"""
import asyncio
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import subprocess

logger = logging.getLogger(__name__)


async def init_artifact_project_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize a new artifact project.
    
    Args:
        params:
            - project_name (str): Name of project
            - output_directory (str, optional): Output directory
            - include_shadcn (bool, optional): Include shadcn/ui
            - include_tailwind (bool, optional): Include Tailwind
    
    Returns:
        Dictionary with project path and files created
    """
    project_name = params.get('project_name', '')
    output_directory = params.get('output_directory', '.')
    include_shadcn = params.get('include_shadcn', True)
    include_tailwind = params.get('include_tailwind', True)
    
    if not project_name:
        return {
            'success': False,
            'error': 'project_name is required'
        }
    
    try:
        output_path = Path(os.path.expanduser(output_directory))
        project_path = output_path / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        files_created = []
        
        # Create package.json
        package_json = project_path / 'package.json'
        package_content = '''{
  "name": "artifact-project",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "bundle": "parcel build index.html --no-source-maps --public-url ./"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "vite": "^5.0.0",
    "parcel": "^2.12.0",
    "typescript": "^5.3.0"
  }
}
'''
        
        if include_tailwind:
            package_content = package_content.replace(
                '"devDependencies": {',
                '"devDependencies": {\n    "tailwindcss": "^3.4.0",\n    "autoprefixer": "^10.4.0",\n    "postcss": "^8.4.0",'
            )
        
        package_json.write_text(package_content, encoding='utf-8')
        files_created.append('package.json')
        
        # Create index.html
        index_html = project_path / 'index.html'
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Artifact</title>
</head>
<body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
</body>
</html>
'''
        
        index_html.write_text(html_content, encoding='utf-8')
        files_created.append('index.html')
        
        # Create src directory
        src_dir = project_path / 'src'
        src_dir.mkdir(exist_ok=True)
        
        # Create main.tsx
        main_tsx = src_dir / 'main.tsx'
        main_content = '''import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
'''
        
        main_tsx.write_text(main_content, encoding='utf-8')
        files_created.append('src/main.tsx')
        
        # Create App.tsx
        app_tsx = src_dir / 'App.tsx'
        app_content = '''import React from 'react';

function App() {
  return (
    <div className="container">
      <h1>Hello, Artifact!</h1>
      <p>Start building your artifact here.</p>
    </div>
  );
}

export default App;
'''
        
        app_tsx.write_text(app_content, encoding='utf-8')
        files_created.append('src/App.tsx')
        
        # Create index.css
        index_css = src_dir / 'index.css'
        css_content = '''* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  line-height: 1.6;
  color: #333;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}
'''
        
        if include_tailwind:
            css_content = '@tailwind base;\n@tailwind components;\n@tailwind utilities;\n\n' + css_content
        
        index_css.write_text(css_content, encoding='utf-8')
        files_created.append('src/index.css')
        
        # Create tsconfig.json
        tsconfig = project_path / 'tsconfig.json'
        tsconfig_content = '''{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
'''
        
        tsconfig.write_text(tsconfig_content, encoding='utf-8')
        files_created.append('tsconfig.json')
        
        # Create vite.config.ts
        vite_config = project_path / 'vite.config.ts'
        vite_content = '''import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
});
'''
        
        vite_config.write_text(vite_content, encoding='utf-8')
        files_created.append('vite.config.ts')
        
        # Create .parcelrc for bundling
        parcelrc = project_path / '.parcelrc'
        parcelrc_content = '''{
  "extends": "@parcel/config-default"
}
'''
        
        parcelrc.write_text(parcelrc_content, encoding='utf-8')
        files_created.append('.parcelrc')
        
        # Create README
        readme = project_path / 'README.md'
        readme_content = f'''# {project_name}

HTML Artifact built with React, TypeScript, and Tailwind CSS.

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Bundle for Artifact

```bash
npm run bundle
```

This creates a single HTML file ready for Claude.ai artifacts.
'''
        
        readme.write_text(readme_content, encoding='utf-8')
        files_created.append('README.md')
        
        next_steps = [
            f"cd {project_name}",
            "npm install",
            "npm run dev  # Start development server",
            "npm run bundle  # Create single HTML artifact"
        ]
        
        return {
            'success': True,
            'project_path': str(project_path),
            'files_created': files_created,
            'next_steps': next_steps
        }
        
    except Exception as e:
        logger.error(f"Artifact project initialization failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def bundle_artifact_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bundle artifact into single HTML file.
    
    Args:
        params:
            - project_path (str): Path to project
            - output_file (str, optional): Output HTML file
    
    Returns:
        Dictionary with bundle path and file size
    """
    project_path = params.get('project_path', '')
    output_file = params.get('output_file', 'bundle.html')
    
    if not project_path:
        return {
            'success': False,
            'error': 'project_path is required'
        }
    
    project_dir = Path(os.path.expanduser(project_path))
    
    if not project_dir.exists():
        return {
            'success': False,
            'error': f'Project directory not found: {project_path}'
        }
    
    try:
        # Check if node_modules exists
        if not (project_dir / 'node_modules').exists():
            return {
                'success': False,
                'error': 'node_modules not found. Run "npm install" first.'
            }
        
        # Run parcel bundle
        bundle_output = project_dir / output_file
        
        # Use shell-exec skill if available
        try:
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
            except ImportError:
                from core.registry.skills_registry import get_skills_registry
            
            registry = get_skills_registry()
            registry.init()
            shell_skill = registry.get_skill('shell-exec')
            
            if shell_skill:
                exec_tool = shell_skill.tools.get('execute_command_tool')
                
                if exec_tool:
                    # Change to project directory and run bundle
                    cmd = f"cd {project_dir} && npm run bundle"
                    
                    if inspect.iscoroutinefunction(exec_tool):
                        result = await exec_tool({'command': cmd})
                    else:
                        result = exec_tool({'command': cmd})
                    
                    if result.get('success'):
                        # Check if bundle was created
                        if bundle_output.exists():
                            file_size = bundle_output.stat().st_size
                            return {
                                'success': True,
                                'bundle_path': str(bundle_output),
                                'file_size': file_size
                            }
        except Exception as e:
            logger.debug(f"Shell execution failed: {e}")
        
        # Fallback: manual instructions
        return {
            'success': False,
            'error': 'Bundling requires npm. Run manually: cd project_path && npm run bundle',
            'instructions': [
                f"cd {project_path}",
                "npm run bundle",
                f"Output will be: {output_file}"
            ]
        }
        
    except Exception as e:
        logger.error(f"Artifact bundling failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }
