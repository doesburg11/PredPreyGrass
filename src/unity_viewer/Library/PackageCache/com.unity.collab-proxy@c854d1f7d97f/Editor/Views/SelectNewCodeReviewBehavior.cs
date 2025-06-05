using Codice.Client.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.Settings;

namespace Unity.PlasticSCM.Editor.Views
{
    internal class SelectNewCodeReviewBehavior
    {
        internal static NewCodeReviewBehavior For(string repServer)
        {
            if (PlasticGui.Plastic.API.IsCloud(repServer))
                return AskUserIfNeeded();

            return NewCodeReviewBehavior.CreateAndOpenInDesktop;
        }

        static NewCodeReviewBehavior AskUserIfNeeded()
        {
            NewCodeReviewBehavior choice = LoadPreferences();
            if (choice != NewCodeReviewBehavior.Ask)
                return choice;

            return AskUserForNewCodeReviewBehavior();
        }

        static NewCodeReviewBehavior AskUserForNewCodeReviewBehavior()
        {
            MultiLinkLabelData dontAksMeAgainContent =
                new MultiLinkLabelData(
                    PlasticLocalization.Name.DontAskMeAgainWithAction.GetString(),
                    PlasticLocalization.Name.OtherOptions.GetString(),
                    OpenPlasticProjectSettings.InOtherFoldout
                );

            bool dontAskMeAgain;
            GuiMessage.GuiMessageResponseButton response
                = GuiMessage.Get().ShowQuestionWithCheckBox(
                    PlasticLocalization.Name.SelectNewCodeReviewBehaviorTitle.GetString(),
                    PlasticLocalization.Name.SelectNewCodeReviewBehaviorExplanation.GetString(),
                    PlasticLocalization.Name.OpenInDesktopApp.GetString(),
                    PlasticLocalization.Name.CancelButton.GetString(),
                    PlasticLocalization.Name.OpenInUnityCloud.GetString(),
                    dontAksMeAgainContent,
                    out dontAskMeAgain
                );

            NewCodeReviewBehavior choice = GetNewCodeReviewBehavior(response);

            if (dontAskMeAgain && choice != NewCodeReviewBehavior.Ask)
                SavePreference(choice);

            return choice;
        }

        static NewCodeReviewBehavior LoadPreferences()
        {
            return PlasticGuiConfig.Get().Configuration.NewCodeReviewBehavior;
        }

        static void SavePreference(NewCodeReviewBehavior choice)
        {
            PlasticGuiConfig plasticGuiConfig = PlasticGuiConfig.Get();
            plasticGuiConfig.Configuration.NewCodeReviewBehavior = choice;
            plasticGuiConfig.Save();
        }

        static NewCodeReviewBehavior GetNewCodeReviewBehavior(
            GuiMessage.GuiMessageResponseButton response)
        {
            switch (response)
            {
                case GuiMessage.GuiMessageResponseButton.Positive:
                    return NewCodeReviewBehavior.CreateAndOpenInDesktop;
                case GuiMessage.GuiMessageResponseButton.Negative:
                    return NewCodeReviewBehavior.RequestFromUnityCloud;
                case GuiMessage.GuiMessageResponseButton.Neutral:
                default:
                    return NewCodeReviewBehavior.Ask;
            }
        }
    }
}
