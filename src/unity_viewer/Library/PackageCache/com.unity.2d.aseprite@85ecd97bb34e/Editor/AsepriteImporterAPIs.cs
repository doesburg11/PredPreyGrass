using System;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    public partial class AsepriteImporter
    {
        /// <summary>
        /// A parsed representation of the Aseprite file.
        /// </summary>
        public AsepriteFile asepriteFile => m_AsepriteFile;

        /// <summary>
        /// How the file should be imported.
        /// </summary>
        public FileImportModes importMode
        {
            get => m_AsepriteImporterSettings.fileImportMode;
            set => m_AsepriteImporterSettings.fileImportMode = value;
        }

        /// <summary>
        /// Which type of texture are we dealing with here.
        /// </summary>
        /// <value>Valid values are TextureImporterType.Default or TextureImporterType.Sprite.</value>
        /// <exception cref="ArgumentException">Exception when non valid values are set.</exception>
        public TextureImporterType textureType
        {
            get => m_TextureImporterSettings.textureType;
            set
            {
                if (value == TextureImporterType.Sprite || value == TextureImporterType.Default)
                {
                    m_TextureImporterSettings.textureType = value;
                    SetDirty();
                }
                else
                    throw new System.ArgumentException("Invalid value. Valid values are TextureImporterType.Sprite or TextureImporterType.Default");
            }
        }

        /// <summary>
        /// Selects Single or Manual import mode for Sprite textures.
        /// </summary>
        /// <value>Valid values are SpriteImportMode.Multiple or SpriteImportMode.Single.</value>
        /// <exception cref="ArgumentException">Exception when non valid values are set.</exception>
        public SpriteImportMode spriteImportMode
        {
            get => (SpriteImportMode)m_TextureImporterSettings.spriteMode;
            set
            {
                if (value == SpriteImportMode.Multiple || value == SpriteImportMode.Single)
                {
                    m_TextureImporterSettings.spriteMode = (int)value;
                    SetDirty();
                }
                else
                    throw new System.ArgumentException("Invalid value. Valid values are SpriteImportMode.Multiple or SpriteImportMode.Single");
            }
        }

        /// <summary>
        /// The number of pixels in the sprite that correspond to one unit in world space.
        /// </summary>
        public float spritePixelsPerUnit
        {
            get => m_TextureImporterSettings.spritePixelsPerUnit;
            set
            {
                var newPpu = Mathf.Max(1f, value);
                m_TextureImporterSettings.spritePixelsPerUnit = newPpu;
                SetDirty();
            }
        }

        /// <summary>
        /// Sets the type of mesh to ge generated for each Sprites.
        /// </summary>
        public SpriteMeshType spriteMeshType
        {
            get => m_TextureImporterSettings.spriteMeshType;
            set
            {
                m_TextureImporterSettings.spriteMeshType = value;
                SetDirty();
            }
        }

        /// <summary>
        /// If enabled, generates a default physics shape from the outline of the Sprite/s when a physics shape has not been set in the Sprite Editor.
        /// </summary>
        public bool generatePhysicsShape
        {
            get => m_GeneratePhysicsShape;
            set
            {
                m_GeneratePhysicsShape = value;
                SetDirty();
            }
        }

        /// <summary>
        /// The number of blank pixels to leave between the edge of the graphic and the mesh.
        /// </summary>
        public uint spriteExtrude
        {
            get => m_TextureImporterSettings.spriteExtrude;
            set
            {
                m_TextureImporterSettings.spriteExtrude = value;
                SetDirty();
            }
        }

        /// <summary>
        /// The canvas size of the source file.
        /// </summary>
        public Vector2 canvasSize => m_CanvasSize;

        /// <summary>
        /// Should include hidden layers from the source file.
        /// </summary>
        public bool includeHiddenLayers
        {
            get => m_AsepriteImporterSettings.importHiddenLayers;
            set => m_AsepriteImporterSettings.importHiddenLayers = value;
        }

        /// <summary>
        /// The import mode for all layers in the file.
        /// </summary>
        public LayerImportModes layerImportMode
        {
            get => m_AsepriteImporterSettings.layerImportMode;
            set => m_AsepriteImporterSettings.layerImportMode = value;
        }

        /// <summary>
        /// The space the Sprite pivots are being calculated.
        /// </summary>
        public PivotSpaces pivotSpace
        {
            get => m_AsepriteImporterSettings.defaultPivotSpace;
            set => m_AsepriteImporterSettings.defaultPivotSpace = value;
        }

        /// <summary>
        /// How a Sprite's graphic rectangle is aligned with its pivot point.
        /// </summary>
        public SpriteAlignment pivotAlignment
        {
            get => m_AsepriteImporterSettings.defaultPivotAlignment;
            set => m_AsepriteImporterSettings.defaultPivotAlignment = value;
        }


        /// <summary>
        /// Normalized position of the custom pivot.
        /// </summary>
        public Vector2 customPivotPosition
        {
            get => m_AsepriteImporterSettings.customPivotPosition;
            set => m_AsepriteImporterSettings.customPivotPosition = value;
        }

        /// <summary>
        /// External padding between each SpriteRect, in pixels.
        /// </summary>
        public uint mosaicPadding
        {
            get => m_AsepriteImporterSettings.mosaicPadding;
            set => m_AsepriteImporterSettings.mosaicPadding = value;
        }

        /// <summary>
        /// Internal padding within each SpriteRect, in pixels.
        /// </summary>
        public uint spritePadding
        {
            get => m_AsepriteImporterSettings.spritePadding;
            set => m_AsepriteImporterSettings.spritePadding = value;
        }

        /// <summary>
        /// Generate a Model Prefab based on the layers of the source asset.
        /// </summary>
        public bool generateModelPrefab
        {
            get => m_AsepriteImporterSettings.generateModelPrefab;
            set => m_AsepriteImporterSettings.generateModelPrefab = value;
        }

        /// <summary>
        /// Add a Sorting Group component to the root of the generated model prefab if it has more than one Sprite Renderer.
        /// </summary>
        public bool addSortingGroup
        {
            get => m_AsepriteImporterSettings.addSortingGroup;
            set
            {
                m_AsepriteImporterSettings.addSortingGroup = value;
                SetDirty();
            }
        }
        
        /// <summary>
        /// Add Shadow Casters to the generated GameObjects with SpriteRenderers.
        /// </summary>
        public bool addShadowCasters
        {
            get => m_AsepriteImporterSettings.addShadowCasters;
            set
            {
                m_AsepriteImporterSettings.addShadowCasters = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Generate Animation Clips based on the frame data of the source asset.
        /// </summary>
        public bool generateAnimationClips
        {
            get => m_AsepriteImporterSettings.generateAnimationClips;
            set => m_AsepriteImporterSettings.generateAnimationClips = value;
        }

        /// <summary>
        /// Events will be generated with their own method name. If disabled, all events will be received by the method `OnAnimationEvent(string)`.
        /// </summary>
        public bool generateIndividualEvents
        {
            get => m_AsepriteImporterSettings.generateIndividualEvents;
            set => m_AsepriteImporterSettings.generateIndividualEvents = value;
        }
        
        /// <summary>
        /// Generate a Sprite Atlas to contain the created texture. This is only available when importing a Tile Set.
        /// </summary>
        public bool generateSpriteAtlas
        {
            get => m_AsepriteImporterSettings.generateSpriteAtlas;
            set => m_AsepriteImporterSettings.generateSpriteAtlas = value;
        }        

        /// <summary>
        /// Texture coordinate wrapping mode.
        /// <br/><br/>Using wrapMode sets the same wrapping mode on all axes. Different per-axis wrap modes can be set using wrapModeU, wrapModeV, wrapModeW. Querying the value returns the U axis wrap mode (same as wrapModeU getter).
        /// </summary>
        public TextureWrapMode wrapMode
        {
            get => m_TextureImporterSettings.wrapMode;
            set
            {
                m_TextureImporterSettings.wrapMode = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Texture U coordinate wrapping mode.
        /// <br/><br/>Controls wrapping mode along texture U (horizontal) axis.
        /// </summary>
        public TextureWrapMode wrapModeU
        {
            get => m_TextureImporterSettings.wrapModeU;
            set
            {
                m_TextureImporterSettings.wrapModeU = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Texture V coordinate wrapping mode.
        /// <br/><br/>Controls wrapping mode along texture V (vertical) axis.
        /// </summary>
        public TextureWrapMode wrapModeV
        {
            get => m_TextureImporterSettings.wrapModeV;
            set
            {
                m_TextureImporterSettings.wrapModeV = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Texture W coordinate wrapping mode for Texture3D.
        /// <br/><br/>Controls wrapping mode along texture W (depth, only relevant for Texture3D) axis.
        /// </summary>
        public TextureWrapMode wrapModeW
        {
            get => m_TextureImporterSettings.wrapModeW;
            set
            {
                m_TextureImporterSettings.wrapModeW = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Filtering mode of the texture.
        /// </summary>
        public FilterMode filterMode
        {
            get => m_TextureImporterSettings.filterMode;
            set
            {
                m_TextureImporterSettings.filterMode = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Anisotropic filtering level of the texture.
        /// </summary>
        public int aniso
        {
            get => m_TextureImporterSettings.aniso;
            set
            {
                m_TextureImporterSettings.aniso = value;
                SetDirty();
            }
        }
        
        /// <summary>
        /// Whether this texture stores data in sRGB (also called gamma) color space.
        /// </summary>
        public bool sRGBTexture
        {
            get => m_TextureImporterSettings.sRGBTexture;
            set
            {
                m_TextureImporterSettings.sRGBTexture = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Mip map bias of the texture.
        /// </summary>
        public float mipMapBias
        {
            get => m_TextureImporterSettings.mipmapBias;
            set
            {
                m_TextureImporterSettings.mipmapBias = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Generate Mip Maps.
        /// <br/><br/>Select this to enable mip-map generation. Mipmaps are smaller versions of the Texture that get used when the Texture is very small on screen.
        /// </summary>
        public bool mipmapEnabled
        {
            get => m_TextureImporterSettings.mipmapEnabled;
            set
            {
                m_TextureImporterSettings.mipmapEnabled = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Mip level where texture is faded out completely.
        /// </summary>
        public int mipmapFadeDistanceEnd
        {
            get => m_TextureImporterSettings.mipmapFadeDistanceEnd;
            set
            {
                m_TextureImporterSettings.mipmapFadeDistanceEnd = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Mip level where texture begins to fade out.
        /// </summary>
        public int mipmapFadeDistanceStart
        {
            get => m_TextureImporterSettings.mipmapFadeDistanceStart;
            set
            {
                m_TextureImporterSettings.mipmapFadeDistanceEnd = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Enable mipmap streaming for the texture.
        /// <br/><br/>Only load larger mipmaps as needed to render the current game cameras. Requires texture streaming to be enabled in quality settings.
        /// </summary>
        public bool streamingMipmaps
        {
            get => m_TextureImporterSettings.streamingMipmaps;
            set
            {
                m_TextureImporterSettings.streamingMipmaps = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Mipmap streaming priority when there's contention for resources. Positive numbers represent higher priority. The valid range is -128 to 127.
        /// </summary>
        public int streamingMipmapsPriority
        {
            get => m_TextureImporterSettings.streamingMipmapsPriority;
            set
            {
                m_TextureImporterSettings.streamingMipmapsPriority = Mathf.Clamp(value, -128, 127);
                SetDirty();
            }
        }

        /// <summary>
        /// Mip level where texture is faded out completely.
        /// </summary>
        public TextureImporterMipFilter mipmapFilter
        {
            get => m_TextureImporterSettings.mipmapFilter;
            set
            {
                m_TextureImporterSettings.mipmapFilter = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Enables or disables coverage-preserving alpha mipmapping.
        /// <br/><br/>Enable this to rescale the alpha values of computed mipmaps so coverage is preserved. This means a higher percentage of pixels passes the alpha test and lower mipmap levels do not become more transparent. This is disabled by default (set to false).
        /// </summary>
        public bool mipMapsPreserveCoverage
        {
            get => m_TextureImporterSettings.mipMapsPreserveCoverage;
            set
            {
                m_TextureImporterSettings.mipMapsPreserveCoverage = value;
                SetDirty();
            }
        }

        /// <summary>
        /// Retrieves the platform settings used by the importer for a given build target.
        /// </summary>
        /// <param name="buildTarget">The build target to query.</param>
        /// <returns>TextureImporterPlatformSettings used for importing the texture for the build target.</returns>
        public TextureImporterPlatformSettings GetImporterPlatformSettings(BuildTarget buildTarget)
        {
            return PlatformSettingsUtilities.GetPlatformTextureSettings(buildTarget, in m_PlatformSettings);
        }

        /// <summary>
        /// Sets the platform settings used by the importer for a given build target.
        /// </summary>
        /// <param name="setting">TextureImporterPlatformSettings to be used by the importer for the build target indicated by TextureImporterPlatformSettings.</param>
        public void SetImporterPlatformSettings(TextureImporterPlatformSettings setting)
        {
            SetPlatformTextureSettings(setting);
            SetDirty();
        }

        /// <summary>
        /// Structure used for Aseprite Import Events.
        /// </summary>
        public readonly struct ImportEventArgs
        {
            /// <summary>
            /// The Aseprite Importer that fired the event.
            /// </summary>
            public readonly AsepriteImporter importer;
            /// <summary>
            /// The Asset Import Context that is being used for the import.
            /// </summary>
            public readonly AssetImportContext context;

            /// <summary>
            /// Constructor for ImportEventArgs.
            /// </summary>
            /// <param name="importer">The Aseprite Importer that fired the event.</param>
            /// <param name="context">The Asset Import Context that is being used for the import.</param>
            public ImportEventArgs(AsepriteImporter importer, AssetImportContext context)
            {
                this.importer = importer;
                this.context = context;
            }
        }

        /// <summary>
        /// Delegate for Aseprite Import Events.
        /// </summary>
        /// <param name="args">The ImportEventArgs that are being used for the import.</param>
        public delegate void AsepriteImportEventHandler(ImportEventArgs args);

        /// <summary>
        /// Event that is fired at the last step of the Aseprite import process.
        /// </summary>
        public AsepriteImportEventHandler OnPostAsepriteImport { get; set; }
    }
}
