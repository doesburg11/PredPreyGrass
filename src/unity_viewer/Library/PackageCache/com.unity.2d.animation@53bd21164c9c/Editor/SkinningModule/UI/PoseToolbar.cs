using System;
using UnityEditor.U2D.Common;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Animation
{
#if ENABLE_UXML_SERIALIZED_DATA
    [UxmlElement]
#endif
    internal partial class PoseToolbar : Toolbar
    {
        private const string k_UxmlPath = "SkinningModule/PoseToolbar.uxml";
        private const string k_UssPath = "SkinningModule/PoseToolbarStyle.uss";
        private const string k_ToolbarId = "PoseToolbar";

        private const string k_PreviewPoseId = "PreviewPose";
        private const string k_RestorePoseId = "RestorePose";
        private const string k_CharacterPivotId = "PivotPose";

#if ENABLE_UXML_TRAITS
        public class CustomUXMLFactor : UxmlFactory<PoseToolbar, UxmlTraits> { }
#endif

        public event Action<Tools> SetMeshTool = (mode) => { };
        public event Action<Tools> SetSkeletonTool = (mode) => { };

        public event Action ActivateEditPoseTool = () => { };

        SkinningCache skinningCache { get; set; }

        private Button m_PreviewBtn;
        private Button m_RestoreBtn;
        private Button m_PivotBtn;

        public static PoseToolbar GenerateFromUXML()
        {
            var clone = GetClone(k_UxmlPath, k_ToolbarId) as PoseToolbar;
            clone.BindElements();
            clone.SetupShortcutUtility();
            clone.LocalizeTextInChildren();
            clone.AddShortcutsToToolTips();
            return clone;
        }

        public PoseToolbar()
        {
            styleSheets.Add(ResourceLoader.Load<StyleSheet>(k_UssPath));
        }

        public void Setup(SkinningCache s)
        {
            skinningCache = s;
            skinningCache.events.skinningModeChanged.AddListener(OnSkinningModeChange);
            OnSkinningModeChange(s.mode);
        }

        private void BindElements()
        {
            m_PreviewBtn = this.Q<Button>(k_PreviewPoseId);
            m_PreviewBtn.clickable.clicked += () => { ActivateEditPoseTool(); };

            m_RestoreBtn = this.Q<Button>(k_RestorePoseId);
            m_RestoreBtn.clickable.clicked += RestorePose;

            m_PivotBtn = this.Q<Button>(k_CharacterPivotId);
            m_PivotBtn.clickable.clicked += PivotPose;
        }

        private void PivotPose()
        {
            SetMeshTool(Tools.CharacterPivotTool);
        }

        private void OnSkinningModeChange(SkinningMode mode)
        {
            if (skinningCache.hasCharacter)
            {
                m_PivotBtn.SetHiddenFromLayout(false);
                if (mode == SkinningMode.SpriteSheet)
                {
                    m_PivotBtn.SetEnabled(false);
                    if (skinningCache.GetTool(Tools.CharacterPivotTool).isActive)
                        SetSkeletonTool(Tools.EditPose);
                }
                else if (mode == SkinningMode.Character)
                {
                    m_PivotBtn.SetEnabled(true);
                }
            }
            else
            {
                m_PivotBtn.SetHiddenFromLayout(true);
                var tool = skinningCache.GetTool(Tools.CharacterPivotTool);
                if (tool != null && tool.isActive)
                    SetSkeletonTool(Tools.EditPose);
            }
        }

        private void RestorePose()
        {
            using (skinningCache.UndoScope(TextContent.restorePose))
            {
                skinningCache.RestoreBindPose();
                skinningCache.events.restoreBindPose.Invoke();
            }
        }

        private void SetupShortcutUtility()
        {
            m_ShortcutUtility = new ShortcutUtility(ShortcutIds.previewPose,
                ShortcutIds.restoreBindPose);
            m_ShortcutUtility.OnShortcutChanged = () =>
            {
                RestoreButtonTooltips(k_UxmlPath, k_ToolbarId);
                AddShortcutsToToolTips();
            };
        }

        public void UpdateToggleState()
        {
            SetButtonChecked(m_PreviewBtn, skinningCache.GetTool(Tools.EditPose).isActive);
            SetButtonChecked(m_PivotBtn, skinningCache.GetTool(Tools.CharacterPivotTool).isActive);
        }

        public void UpdateResetButtonState()
        {
            var skeleton = skinningCache.GetEffectiveSkeleton(skinningCache.selectedSprite);
            var isResetEnabled = skeleton != null && skeleton.isPosePreview;
            m_RestoreBtn.SetEnabled(isResetEnabled);
        }

        private void AddShortcutsToToolTips()
        {
            m_ShortcutUtility.AddShortcutToButtonTooltip(this, k_PreviewPoseId, ShortcutIds.previewPose);
            m_ShortcutUtility.AddShortcutToButtonTooltip(this, k_RestorePoseId, ShortcutIds.restoreBindPose);
            m_ShortcutUtility.AddShortcutToButtonTooltip(this, k_CharacterPivotId, ShortcutIds.characterPivot);
        }
    }
}
