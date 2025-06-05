using System.Diagnostics;
using NUnit.Framework.Interfaces;
using NUnit.Framework.Internal;
using NUnit.Framework.Internal.Commands;

namespace Tests_16s
{
    public class FasterThan500msCommand : TestCommand
    {
        private TestCommand innerCommand;
        
        public FasterThan500msCommand(TestCommand command) : base(command.Test)
        {
            innerCommand = command;
        }

        public override TestResult Execute(ITestExecutionContext context)
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var result = innerCommand.Execute(context);
            stopWatch.Stop();

            if (stopWatch.ElapsedMilliseconds > 500)
            {
                result.SetResult(ResultState.Failure, $"Test took {stopWatch.ElapsedMilliseconds} ms. That is longer than 500ms!");
            }

            return result;
        }
    }
}