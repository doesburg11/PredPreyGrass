using UnityEditor.Overlays;
using UnityEngine;
using UnityEngine.UIElements;

namespace UnityEditor.Tilemaps
{
    internal abstract class BoolFieldOverlayPopupWindow : OverlayPopupWindow
    {
        private BaseField<bool> trigger;
        private bool isLocked;
        private Rect screenRect;
        private Vector2 size;

        protected virtual void OnDisable()
        {
            // Disable trigger state if closed outside of trigger
            if (trigger != null && (trigger.pseudoStates & PseudoStates.Hover) == 0)
                trigger.SetValueWithoutNotify(false);
        }

        protected void ToggleMode<T>(bool value) where T : BoolFieldOverlayPopupWindow
        {
            isLocked = value;
            ShowOverlayPopup<T>(trigger, screenRect, size, isLocked);;
        }

        private void InitDropDown()
        {
            var mode = ShowMode.PopupMenu;
            var giveFocus = true;

            this.position = this.ShowAsDropDownFitToScreen(screenRect, size, null);
            if (ContainerWindow.IsPopup(mode))
                this.ShowPopupWithMode(mode, giveFocus);
            else
                this.ShowWithMode(mode);

            this.position = this.ShowAsDropDownFitToScreen(screenRect, size, null);
            double width1 = (double)this.position.width;
            Rect position = this.position;
            double height1 = (double)position.height;
            this.minSize = new Vector2((float)width1, (float)height1);
            position = this.position;
            double width2 = (double)position.width;
            position = this.position;
            double height2 = (double)position.height;
            this.maxSize = new Vector2((float)width2, (float)height2);

            if (giveFocus && (UnityEngine.Object)EditorWindow.focusedWindow != (UnityEngine.Object) this)
                this.Focus();
            else
                this.Repaint();

            if (!isLocked)
                this.m_Parent.AddToAuxWindowList();
            this.m_Parent.window.m_DontSaveToLayout = true;
        }

        public static void ShowOverlayPopup<T>(BaseField<bool> trigger, Rect position, Vector2 size, bool isLocked) where T : BoolFieldOverlayPopupWindow
        {
            CloseAllWindows<T>();
            var instance = ScriptableObject.CreateInstance<T>();
            instance.trigger = trigger;
            instance.size = size;
            instance.screenRect = position;
            instance.isLocked = isLocked;
            instance.InitDropDown();
        }

        public static void ShowOverlayPopup<T>(BaseField<bool> trigger, Vector2 size, bool isLocked) where T : BoolFieldOverlayPopupWindow
        {
            var rect = GUIUtility.GUIToScreenRect(trigger.worldBound);
            ShowOverlayPopup<T>(trigger, rect, size, isLocked);
        }

        public static void CloseAllWindows<T>() where T : BoolFieldOverlayPopupWindow
        {
            foreach (T obj in Resources.FindObjectsOfTypeAll<T>())
                obj.Close();
        }
    }
}
