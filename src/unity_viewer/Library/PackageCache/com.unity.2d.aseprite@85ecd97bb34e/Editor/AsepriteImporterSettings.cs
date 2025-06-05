using System;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    [Serializable]
    internal struct AsepriteImporterSettings
    {
        [SerializeField] FileImportModes m_FileImportMode;
        public FileImportModes fileImportMode
        {
            get => m_FileImportMode;
            set => m_FileImportMode = value;
        }

        [SerializeField] bool m_ImportHiddenLayers;
        public bool importHiddenLayers
        {
            get => m_ImportHiddenLayers;
            set => m_ImportHiddenLayers = value;
        }

        [SerializeField] LayerImportModes m_LayerImportMode;
        public LayerImportModes layerImportMode
        {
            get => m_LayerImportMode;
            set => m_LayerImportMode = value;
        }

        [SerializeField] PivotSpaces m_DefaultPivotSpace;
        public PivotSpaces defaultPivotSpace
        {
            get => m_DefaultPivotSpace;
            set => m_DefaultPivotSpace = value;
        }

        [SerializeField] SpriteAlignment m_DefaultPivotAlignment;
        public SpriteAlignment defaultPivotAlignment
        {
            get => m_DefaultPivotAlignment;
            set => m_DefaultPivotAlignment = value;
        }

        [SerializeField] Vector2 m_CustomPivotPosition;
        public Vector2 customPivotPosition
        {
            get => m_CustomPivotPosition;
            set => m_CustomPivotPosition = value;
        }

        [SerializeField] uint m_MosaicPadding;
        public uint mosaicPadding
        {
            get => m_MosaicPadding;
            set => m_MosaicPadding = value;
        }

        [SerializeField] uint m_SpritePadding;
        public uint spritePadding
        {
            get => m_SpritePadding;
            set => m_SpritePadding = value;
        }

        [SerializeField] bool m_GenerateModelPrefab;
        public bool generateModelPrefab
        {
            get => m_GenerateModelPrefab;
            set => m_GenerateModelPrefab = value;
        }

        [SerializeField] bool m_GenerateAnimationClips;
        public bool generateAnimationClips
        {
            get => m_GenerateAnimationClips;
            set => m_GenerateAnimationClips = value;
        }

        [SerializeField] bool m_AddSortingGroup;
        public bool addSortingGroup
        {
            get => m_AddSortingGroup;
            set => m_AddSortingGroup = value;
        }

        [SerializeField] bool m_AddShadowCasters;
        public bool addShadowCasters
        {
            get => m_AddShadowCasters;
            set => m_AddShadowCasters = value;
        }

        [SerializeField] bool m_GenerateIndividualEvents;
        public bool generateIndividualEvents
        {
            get => m_GenerateIndividualEvents;
            set => m_GenerateIndividualEvents = value;
        }
        
        [SerializeField] bool m_GenerateSpriteAtlas;
        public bool generateSpriteAtlas
        {
            get => m_GenerateSpriteAtlas;
            set => m_GenerateSpriteAtlas = value;
        }        

        public bool IsDefault()
        {
            return !m_ImportHiddenLayers &&
                   m_LayerImportMode == 0 &&
                   m_DefaultPivotSpace == 0 &&
                   m_DefaultPivotAlignment == 0 &&
                   !m_GenerateModelPrefab &&
                   !m_GenerateAnimationClips &&
                   !m_AddSortingGroup &&
                   !m_AddShadowCasters &&
                   !m_GenerateIndividualEvents &&
                   !m_GenerateSpriteAtlas;
        }
    }
}
