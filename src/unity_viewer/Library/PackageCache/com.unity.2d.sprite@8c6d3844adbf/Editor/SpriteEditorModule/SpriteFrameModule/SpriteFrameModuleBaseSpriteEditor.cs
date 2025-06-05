using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.U2D.Sprites
{
    internal abstract partial class SpriteFrameModuleBase : SpriteEditorModuleModeSupportBase, ISpriteEditor
    {

        public override ISpriteEditor modeSpriteEditor => this;
        /// <summary>
        /// Mimics ISpriteEditor for SpriteEditorMode. Calls are route back to SpriteEditor
        /// but can be override by if SpriteEditorModuleModeSupportBase it wants specific behaviour.
        /// </summary>
        #region ISpriteEditor implementation.

        public virtual List<SpriteRect> spriteRects
        {
            set => spriteEditor.spriteRects = value;
        }

        public virtual SpriteRect selectedSpriteRect
        {
            get => spriteEditor.selectedSpriteRect;
            set => spriteEditor.selectedSpriteRect = value;
        }
        public virtual bool enableMouseMoveEvent { set => spriteEditor.enableMouseMoveEvent = value; }
        public virtual bool editingDisabled => spriteEditor.editingDisabled;
        public virtual Rect windowDimension => spriteEditor.windowDimension;

        public virtual T GetDataProvider<T>() where T : class
        {
            return spriteEditor.GetDataProvider<T>();
        }

        public virtual bool HandleSpriteSelection()
        {
            return spriteEditor.HandleSpriteSelection();
        }

        public virtual void RequestRepaint()
        {
            spriteEditor.RequestRepaint();
        }

        public virtual void SetDataModified()
        {
            spriteEditor.SetDataModified();
        }

        public virtual void ApplyOrRevertModification(bool apply)
        {
            spriteEditor.ApplyOrRevertModification(apply);
        }

        public virtual VisualElement GetMainVisualContainer()
        {
            return spriteEditor.GetMainVisualContainer();
        }

        public virtual  VisualElement GetToolbarRootElement()
        {
            return spriteEditor.GetToolbarRootElement();
        }

        public virtual void SetPreviewTexture(Texture2D texture, int width, int height)
        {
            spriteEditor.SetPreviewTexture(texture, width, height);
        }

        public virtual void ResetZoomAndScroll()
        {
            spriteEditor.ResetZoomAndScroll();
        }

        public virtual float zoomLevel
        {
            get => spriteEditor.zoomLevel;
            set => spriteEditor.zoomLevel = value;

        }
        public virtual Vector2 scrollPosition
        {
            get => spriteEditor.scrollPosition;
            set => spriteEditor.scrollPosition = value;
        }

        public virtual bool showAlpha
        {
            get => spriteEditor.showAlpha;
            set => spriteEditor.showAlpha = value;
        }

        public virtual float mipLevel
        {
            get => spriteEditor.mipLevel;
            set => spriteEditor.mipLevel = value;
        }

        #endregion
    }
}
