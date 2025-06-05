using System;
using UnityEditor.Tilemaps.External;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal class TilePaletteBrushPickItemElement : VisualElement
    {
        private static string kUssClassName = "unity-tilepalette-brushpick-item";
        private static string kIconUssClassName = "unity-tilepalette-brushpick-item__icon";

        private static string kInvalidBrushText =
            L10n.Tr("Invalid Brush. Has the Brush script been removed from the project?");

        private GridBrushBase m_Brush;
        private Image m_Image;
        private VisualElement m_Icon;
        private RenameableLabel m_BrushName;

        public Action pointerUpEvent;
        public Action<string> renameEvent;

        public TilePaletteBrushPickItemElement(bool hasName)
        {
            AddToClassList(kUssClassName);

            m_Image = new Image();
            m_Image.RegisterCallback<PointerUpEvent>(evt => pointerUpEvent?.Invoke());
            Add(m_Image);

            m_Icon = new VisualElement();
            m_Icon.AddToClassList(kIconUssClassName);
            m_Image.Add(m_Icon);

            if (hasName)
            {
                m_BrushName = new RenameableLabel();
                Add(m_BrushName);
                m_BrushName.renameEnding += BrushNameOnRenameEnding;
            }
        }

        private void BrushNameOnRenameEnding(RenameableLabel label, bool isCancelled)
        {
            if (isCancelled)
                return;

            renameEvent?.Invoke(label.text);
        }

        private void SetAssetPreview()
        {
            if (m_Brush == null)
            {
                m_Image.image = null;
                return;
            }

            var id = m_Brush.GetInstanceID();
            var assetPreview = AssetPreview.GetAssetPreview(m_Brush);
            if (assetPreview == null
                || AssetPreview.IsLoadingAssetPreview(id)
                || !AssetPreview.HasAssetPreview(id, 0))
            {
                schedule.Execute(SetAssetPreview).StartingIn(300);
            }
            else
            {
                m_Image.image = assetPreview;
            }
        }

        public void SetBrush(GridBrushBase brush)
        {
            if (brush == null)
            {
                m_Image.image = null;
                m_Icon.style.backgroundImage = EditorGUIUtility.LoadIconRequired("console.warnicon.sml");
                m_Icon.tooltip = kInvalidBrushText;
                if (m_BrushName != null)
                {
                    m_BrushName.visible = false;
                    m_BrushName.text = null;
                    m_BrushName.tooltip = null;
                }
            }
            else
            {
                m_Brush = brush;
                SetAssetPreview();
                m_Icon.style.backgroundImage = GridPaletteBrushes.GetBrushIcon(brush.GetType());
                m_Icon.tooltip = ObjectNames.NicifyVariableName(brush.GetType().Name);
                if (m_BrushName != null)
                {
                    m_BrushName.visible = visible;
                    m_BrushName.text = brush.name;
                    m_BrushName.tooltip = brush.name;
                }
            }
        }

        public void SetSize(int size)
        {
            m_Image.style.width = size;
            m_Image.style.height = size;
        }

        public void ToggleRename()
        {
            if (m_BrushName.isRenaming)
                m_BrushName.CancelRename();
            else
                m_BrushName.BeginRename();
        }
    }

    internal class TilePaletteBrushPickTypeElement : VisualElement
    {
        private static string kUssClassName = "unity-tilepalette-brushpick-type";
        private static string kIconUssClassName = "unity-tilepalette-brushpick-type__icon";

        private static string kInvalidBrushText =
            L10n.Tr("Invalid Brush. Has the Brush script been removed from the project?");

        private VisualElement m_Icon;
        private Label m_Label;

        public Action pointerUpEvent;
        public Action<string> renameEvent;

        public TilePaletteBrushPickTypeElement()
        {
            AddToClassList(kUssClassName);

            m_Icon = new VisualElement();
            m_Icon.AddToClassList(kIconUssClassName);
            Add(m_Icon);

            m_Label = new Label();
            Add(m_Label);
        }

        public void SetBrush(GridBrushBase brush)
        {
            if (brush == null)
            {
                m_Icon.style.backgroundImage = EditorGUIUtility.LoadIconRequired("console.warnicon.sml");
                m_Icon.tooltip = kInvalidBrushText;
                m_Label.text = null;
            }
            else
            {
                var name = ObjectNames.NicifyVariableName(brush.GetType().Name);
                m_Icon.style.backgroundImage = GridPaletteBrushes.GetBrushIcon(brush.GetType());
                m_Icon.tooltip = name;
                m_Label.text = name;
            }
        }
    }
}
