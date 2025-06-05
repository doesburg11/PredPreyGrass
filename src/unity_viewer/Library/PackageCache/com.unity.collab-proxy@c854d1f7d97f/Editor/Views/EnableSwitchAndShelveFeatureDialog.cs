using System.Threading.Tasks;

using UnityEditor;
using UnityEngine;

using Codice.Client.Common.EventTracking;
using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Views
{
    internal class EnableSwitchAndShelveFeature :
        SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog
    {
        internal EnableSwitchAndShelveFeature(RepositorySpec repSpec, EditorWindow window)
        {
            mRepSpec = repSpec;
            mWindow = window;
        }

        bool SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog.Show()
        {
            bool result = false;

            GUIActionRunner.RunGUIAction(() =>
            {
                result = EnableSwitchAndShelveFeatureDialog.Show(mRepSpec, mWindow);
            });

            return result;
        }

        readonly EditorWindow mWindow;
        readonly RepositorySpec mRepSpec;
    }

    internal class EnableSwitchAndShelveFeatureDialog : PlasticDialog
    {
        protected override Rect DefaultRect
        {
            get
            {
                var baseRect = base.DefaultRect;
                return new Rect(baseRect.x, baseRect.y, 600, 320);
            }
        }

        internal static bool Show(RepositorySpec repSpec, EditorWindow window)
        {
            EnableSwitchAndShelveFeatureDialog dialog = CreateInstance<EnableSwitchAndShelveFeatureDialog>();
            dialog.mRepSpec = repSpec;
            ResponseType dialogResult = dialog.RunModal(window);
            return dialogResult == ResponseType.Ok;
        }

        protected override string GetTitle()
        {
            return PlasticLocalization.Name.EnableSwitchAndShelveTitle.GetString();
        }

        protected override void OnModalGUI()
        {
            Title(PlasticLocalization.Name.EnableSwitchAndShelveTitle.GetString());

            Paragraph(PlasticLocalization.Name.EnableSwitchAndShelveMessage.GetString());

            using (new EditorGUILayout.HorizontalScope())
            {
                EditorGUILayout.Space(20);
                using (new EditorGUILayout.VerticalScope())
                {
                    Paragraph(string.Concat(
                        PlasticLocalization.Name.EnableSwitchAndShelveLeaveChangesTitle.GetString(), "\n",
                        PlasticLocalization.Name.EnableSwitchAndShelveLeaveChangesDescription.GetString()));

                    Paragraph(string.Concat(
                        PlasticLocalization.Name.EnableSwitchAndShelveBringChangesTitle.GetString(), "\n",
                        PlasticLocalization.Name.EnableSwitchAndShelveBringChangesDescription.GetString()));
                }

                GUILayout.FlexibleSpace();
            }

            Paragraph(string.Concat(
                PlasticLocalization.Name.EnableSwitchAndShelveQuestionStart.GetString(), "\n",
                PlasticLocalization.Name.EnableSwitchAndShelveQuestionEnd.GetString()));

            GUILayout.Space(20);

            DoButtonsArea();
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.FlexibleSpace();

                if (Application.platform == RuntimePlatform.WindowsEditor)
                {
                    DoYesButton();
                    DoNoButton();
                    return;
                }

                DoNoButton();
                DoYesButton();
            }
        }

        void DoYesButton()
        {
            if (!NormalButton(PlasticLocalization.Name.EnableSwitchAndShelveYesEnableItLowerCase.GetString()))
                return;

            TrackFeatureUseEvent.For(
                mRepSpec,
                TrackFeatureUseEvent.Features.SwitchAndShelve.EnableFeatureYes);

            OkButtonAction();
        }

        void DoNoButton()
        {
            if (!NormalButton(PlasticLocalization.Name.EnableSwitchAndShelveNotNow.GetString()))
                return;

            TrackFeatureUseEvent.For(
                mRepSpec,
                TrackFeatureUseEvent.Features.SwitchAndShelve.EnableFeatureNo);

            CancelButtonAction();
        }

        RepositorySpec mRepSpec;
    }
}
