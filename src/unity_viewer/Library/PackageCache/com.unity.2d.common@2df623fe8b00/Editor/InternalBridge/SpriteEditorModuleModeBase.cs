#if ENABLE_SPRITEMODULE_MODE
using System;
using System.Collections.Generic;
using UnityEditor.U2D.Sprites;
using UnityEngine;

namespace UnityEditor.U2D.Common
{
    internal abstract class SpriteEditorModeBase : UnityEditor.U2D.Sprites.SpriteEditorModeBase
    {
        event Action<Sprites.SpriteEditorModeBase> m_ModeActivateCallback = _ => { };
        event Action<SpriteRect> m_SpriteEditorSpriteSelectionChanged = _ => { };
        SpriteEditorModuleModeSupportBase m_Module;

        public SpriteEditorModuleBase module => m_Module;

        public override bool ActivateMode()
        {
            return false;
        }

        public override void DeactivateMode()
        { }

        public override void OnAddToModule(UnityEditor.U2D.Sprites.SpriteEditorModuleModeSupportBase module)
        {
            m_Module = module;
            spriteEditor.GetMainVisualContainer().RegisterCallback<SpriteSelectionChangeEvent>(OnSpriteEditorSpriteSelectionChanged);
            OnAddToModuleInternal(module);
        }

        protected abstract void OnAddToModuleInternal(SpriteEditorModuleBase module);

        public override void OnRemoveFromModule(UnityEditor.U2D.Sprites.SpriteEditorModuleModeSupportBase module)
        {
            if (m_Module == module)
            {
                spriteEditor.GetMainVisualContainer().UnregisterCallback<SpriteSelectionChangeEvent>(OnSpriteEditorSpriteSelectionChanged);
                OnRemoveFromModuleInternal(module);
                m_Module = null;
            }
        }

        void OnSpriteEditorSpriteSelectionChanged(SpriteSelectionChangeEvent evt)
        {
            m_SpriteEditorSpriteSelectionChanged?.Invoke(spriteEditor.selectedSpriteRect);
        }

        protected abstract void OnRemoveFromModuleInternal(SpriteEditorModuleBase module);

        public override void RegisterOnModeRequestActivate(Action<Sprites.SpriteEditorModeBase> onActivate)
        {
            m_ModeActivateCallback += onActivate;
        }

        public override void UnregisterOnModeRequestActivate(Action<Sprites.SpriteEditorModeBase> onActivate)
        {
            m_ModeActivateCallback -= onActivate;
        }

        protected void SignalModeActivate(SpriteEditorModeBase mode)
        {
            m_ModeActivateCallback(mode);
        }

        public override bool ApplyModeData(bool apply, HashSet<Type> dataProviderTypes)
        {
            return apply;
        }

        public void RegisterModuleActivate(Action onActivate)
        {
            m_Module.RegisterModuleActivate(onActivate);
        }

        public void UnregisterModuleActivate(Action onActivate)
        {
            m_Module.UnregisterModuleActivate(onActivate);
        }

        public void RegisterSpriteEditorSpriteSelectionChanged(Action<SpriteRect> onSpriteRectChanged)
        {
            m_SpriteEditorSpriteSelectionChanged += onSpriteRectChanged;
        }

        public void UnregisterSpriteEditorSpriteSelectionChanged(Action<SpriteRect> onSpriteRectChanged)
        {
            m_SpriteEditorSpriteSelectionChanged -= onSpriteRectChanged;
        }
    }
}
#endif