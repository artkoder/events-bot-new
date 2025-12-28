"""
Font Manager for 3D Intro Blender scenes.

Centralized system for loading and managing custom fonts with fallback support.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Base directories
_3D_INTRO_DIR = Path(__file__).parent
_ASSETS_DIR = _3D_INTRO_DIR / "assets"
_FONTS_DIR = _ASSETS_DIR / "fonts"
_VIDEO_ANNOUNCE_FONTS = Path(__file__).parent.parent / "video_announce" / "assets"

# Font identifiers
DRUK_CYR_BOLD = "druk_cyr_bold"
BENZIN_BOLD = "benzin_bold"
BEBAS_NEUE_REGULAR = "bebas_neue_regular"
BEBAS_NEUE_PRO_MIDDLE = "bebas_neue_pro_middle"
BEBAS_NEUE_BOLD = "bebas_neue_bold"


class FontManager:
    """Manages font loading and fallback for Blender scenes."""
    
    # Font mapping: font_id -> possible filenames (in priority order)
    FONT_REGISTRY = {
        DRUK_CYR_BOLD: [
            "DrukCyr-Bold.ttf",
            "druk-cyr-bold.ttf",
            "Druk-Cyr-Bold.otf",
            "DrukCyrBold.ttf",
        ],
        BENZIN_BOLD: [
            "Benzin-Bold.ttf",
            "benzin-bold.ttf",
            "BenzinBold.ttf",
            "Benzin-Bold.otf",
        ],
        BEBAS_NEUE_REGULAR: [
            "BebasNeue-Regular.ttf",
            "bebas-neue-regular.ttf",
            "BebasNeue.ttf",
        ],
        BEBAS_NEUE_PRO_MIDDLE: [
            "BebasNeuePro-Middle-Regular.ttf",
            "Bebas-Neue-Pro-Middle-Regular.ttf",
            "BebasNeuePro-Regular.ttf",
            # Fallback to Semi-Expanded variant if Middle not found
            "Bebas-Neue-Pro-SemiExpanded-Bold-BF66cf3d78c1f4e.ttf",
        ],
        BEBAS_NEUE_BOLD: [
            "BebasNeue-Bold.ttf",
            "bebas-neue-bold.ttf",
        ],
    }
    
    # Fallback chain: if font not found, use this instead
    FALLBACK_CHAIN = {
        DRUK_CYR_BOLD: BEBAS_NEUE_BOLD,
        BENZIN_BOLD: BEBAS_NEUE_BOLD,
        BEBAS_NEUE_PRO_MIDDLE: BEBAS_NEUE_BOLD,
    }
    
    def __init__(self):
        """Initialize font manager."""
        self._font_cache: dict[str, Path] = {}
        self._search_paths = [
            _FONTS_DIR,
            _VIDEO_ANNOUNCE_FONTS,
        ]
        self._scan_fonts()
    
    def _scan_fonts(self) -> None:
        """Scan search paths and build font cache."""
        for font_id, filenames in self.FONT_REGISTRY.items():
            for search_path in self._search_paths:
                if not search_path.exists():
                    continue
                    
                for filename in filenames:
                    font_path = search_path / filename
                    if font_path.exists():
                        self._font_cache[font_id] = font_path
                        break  # Found font, stop searching variants
                
                if font_id in self._font_cache:
                    break  # Found in this search path, move to next font
    
    def get_font(self, font_id: str, allow_fallback: bool = True) -> Optional[Path]:
        """
        Get path to font file.
        
        Args:
            font_id: Font identifier (e.g., DRUK_CYR_BOLD)
            allow_fallback: If True, return fallback font if primary not found
            
        Returns:
            Path to font file or None if not found
        """
        # Check cache first
        if font_id in self._font_cache:
            return self._font_cache[font_id]
        
        # Try fallback
        if allow_fallback and font_id in self.FALLBACK_CHAIN:
            fallback_id = self.FALLBACK_CHAIN[font_id]
            if fallback_id in self._font_cache:
                print(f"Warning: Font '{font_id}' not found, using fallback '{fallback_id}'")
                return self._font_cache[fallback_id]
        
        return None
    
    def get_font_str(self, font_id: str, allow_fallback: bool = True) -> str:
        """
        Get path to font file as string (for Blender API).
        
        Args:
            font_id: Font identifier
            allow_fallback: If True, return fallback font if primary not found
            
        Returns:
            Absolute path string
            
        Raises:
            ValueError: If font not found and no fallback available
        """
        font_path = self.get_font(font_id, allow_fallback)
        if font_path is None:
            raise ValueError(
                f"Font '{font_id}' not found and no fallback available. "
                f"Please add font files to {_FONTS_DIR}"
            )
        return str(font_path.absolute())
    
    def validate_fonts(self, required_fonts: list[str]) -> dict[str, bool]:
        """
        Validate that required fonts are available.
        
        Args:
            required_fonts: List of font IDs to check
            
        Returns:
            Dict mapping font_id to availability (True/False)
        """
        return {
            font_id: (self.get_font(font_id, allow_fallback=False) is not None)
            for font_id in required_fonts
        }
    
    def list_available_fonts(self) -> dict[str, Path]:
        """Get all available fonts in cache."""
        return self._font_cache.copy()
    
    def get_font_info(self) -> str:
        """Get human-readable info about font availability."""
        lines = ["Font Manager Status:"]
        lines.append(f"Search paths: {[str(p) for p in self._search_paths]}")
        lines.append("\nAvailable fonts:")
        
        for font_id in self.FONT_REGISTRY.keys():
            path = self.get_font(font_id, allow_fallback=False)
            if path:
                lines.append(f"  ✓ {font_id}: {path.name}")
            else:
                fallback = self.get_font(font_id, allow_fallback=True)
                if fallback:
                    fallback_id = self.FALLBACK_CHAIN.get(font_id, "unknown")
                    lines.append(f"  ⚠ {font_id}: MISSING (using fallback: {fallback_id})")
                else:
                    lines.append(f"  ✗ {font_id}: NOT FOUND")
        
        return "\n".join(lines)


# Global singleton instance
_font_manager: Optional[FontManager] = None


def get_font_manager() -> FontManager:
    """Get global FontManager instance."""
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager


# Convenience functions
def get_font(font_id: str, allow_fallback: bool = True) -> Optional[Path]:
    """Get font path. See FontManager.get_font() for details."""
    return get_font_manager().get_font(font_id, allow_fallback)


def get_font_str(font_id: str, allow_fallback: bool = True) -> str:
    """Get font path as string. See FontManager.get_font_str() for details."""
    return get_font_manager().get_font_str(font_id, allow_fallback)


def validate_fonts(required_fonts: list[str]) -> dict[str, bool]:
    """Validate fonts. See FontManager.validate_fonts() for details."""
    return get_font_manager().validate_fonts(required_fonts)


def print_font_info() -> None:
    """Print font availability info."""
    print(get_font_manager().get_font_info())


if __name__ == "__main__":
    # Test font manager
    print_font_info()
