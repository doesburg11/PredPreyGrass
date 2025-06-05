using Codice.Client.Commands;
using Codice.Client.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Views
{
    internal class ApplyShelveWithConflictsQuestionerBuilder :
        IApplyShelveWithConflictsQuestionerBuilder
    {
        public IApplyShelveWithConflictsQuestioner Get()
        {
            return new ApplyShelveWithConflictsQuestioner();
        }
    }

    internal class ApplyShelveWithConflictsQuestioner : IApplyShelveWithConflictsQuestioner
    {
        ApplyShelveWithConflictsAction IApplyShelveWithConflictsQuestioner.ApplyShelveWithConflicts(
            string dstObject, long shelvesetId)
        {
            ApplyShelveWithConflictsAction result = ApplyShelveWithConflictsAction.Cancel;

            GUIActionRunner.RunGUIAction(() =>
            {
                result = ConfirmContinue(dstObject);
            });

            return result;
        }

        static ApplyShelveWithConflictsAction ConfirmContinue(string dstObject)
        {
            GuiMessage.GuiMessageResponseButton result = GuiMessage.ShowQuestion(
                PlasticLocalization.Name.ApplyShelveWithConflictsTitle.GetString(),
                PlasticLocalization.Name.ApplyShelveWithConflictsExplanation.GetString(dstObject),
                PlasticLocalization.Name.ResolveConflictsNow.GetString(),
                PlasticLocalization.Name.ResolveConflictsLater.GetString(),
                null);

            return result == GuiMessage.GuiMessageResponseButton.Positive
                ? ApplyShelveWithConflictsAction.ApplyShelve
                : ApplyShelveWithConflictsAction.Cancel;
        }
    }
}
