using System;
using System.Collections.Generic;
using UnityEngine;
using Object = UnityEngine.Object;

namespace UnityEditor.U2D.Sprites
{
    [RequireSpriteDataProvider(typeof(ISpritePhysicsOutlineDataProvider), typeof(ITextureDataProvider))]
    internal class SpritePhysicsShapeModule : SpriteOutlineModule
    {
        private const float kDefaultPhysicsTessellationDetail = 0.25f;

        public SpritePhysicsShapeModule(ISpriteEditor sem, IEventSystem ege, IUndoSystem us, IAssetDatabase ad, IGUIUtility gu, IShapeEditorFactory sef, ITexture2D outlineTexture)
            : base(sem, ege, us, ad, gu, sef, outlineTexture)
        {
            spriteEditorWindow = sem;
        }

        public override string moduleName
        {
            get { return "Custom Physics Shape"; }
        }

        private ISpriteEditor spriteEditorWindow
        {
            get; set;
        }

        protected override string alterateLabelText => L10n.Tr("From Outline");

        protected override List<Vector2[]> GetAlternateOutlines(GUID spriteID)
        {
            var alternateOutlineProvider = spriteEditorWindow.GetDataProvider<ISpriteOutlineDataProvider>();
            return alternateOutlineProvider.GetOutlines(spriteID);
        }

        public override bool ApplyRevert(bool apply)
        {
            if (m_Outline != null)
            {
                if (apply)
                {
                    var dp = spriteEditorWindow.GetDataProvider<ISpritePhysicsOutlineDataProvider>();
                    for (int i = 0; i < m_Outline.Count; ++i)
                    {
                        dp.SetOutlines(m_Outline[i].spriteID, m_Outline[i].ToListVector());
                        dp.SetTessellationDetail(m_Outline[i].spriteID, m_Outline[i].tessellationDetail);
                    }
                }

                Object.DestroyImmediate(m_Outline);
                m_Outline = null;
            }

            return true;
        }

        protected override int alphaTolerance
        {
            get => SpriteOutlineModulePreference.physicsAlphaTolerance;
            set => SpriteOutlineModulePreference.physicsAlphaTolerance = value;
        }

        protected override void LoadOutline()
        {
            m_Outline = ScriptableObject.CreateInstance<SpriteOutlineModel>();
            m_Outline.hideFlags = HideFlags.HideAndDontSave;
            var spriteDataProvider = spriteEditorWindow.GetDataProvider<ISpriteEditorDataProvider>();
            var outlineDataProvider = spriteEditorWindow.GetDataProvider<ISpritePhysicsOutlineDataProvider>();
            foreach (var rect in spriteDataProvider.GetSpriteRects())
            {
                var outlines = outlineDataProvider.GetOutlines(rect.spriteID);
                m_Outline.AddListVector2(rect.spriteID, outlines);
                m_Outline[m_Outline.Count - 1].tessellationDetail = outlineDataProvider.GetTessellationDetail(rect.spriteID);
            }
        }

        protected override void SetupShapeEditorOutline(SpriteRect spriteRect)
        {
            var physicsShape = m_Outline[spriteRect.spriteID];
            var physicsShapes = GenerateSpriteRectOutline(spriteRect.rect,
                Math.Abs(physicsShape.tessellationDetail - (-1f)) < Mathf.Epsilon ? kDefaultPhysicsTessellationDetail : physicsShape.tessellationDetail,
                (byte) alphaTolerance, m_TextureDataProvider, m_SpriteOutlineToolElement.optimizeOutline);
            m_Outline[spriteRect.spriteID].spriteOutlines = physicsShapes;
            spriteEditorWindow.SetDataModified();
        }
    }
}
