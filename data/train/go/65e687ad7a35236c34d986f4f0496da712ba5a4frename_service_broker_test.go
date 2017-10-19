package servicebroker_test

import (
	testapi "github.com/nttlabs/cli/cf/api/fakes"
	. "github.com/nttlabs/cli/cf/commands/servicebroker"
	"github.com/nttlabs/cli/cf/configuration/core_config"
	"github.com/nttlabs/cli/cf/models"
	testcmd "github.com/nttlabs/cli/testhelpers/commands"
	testconfig "github.com/nttlabs/cli/testhelpers/configuration"
	. "github.com/nttlabs/cli/testhelpers/matchers"
	testreq "github.com/nttlabs/cli/testhelpers/requirements"
	testterm "github.com/nttlabs/cli/testhelpers/terminal"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("rename-service-broker command", func() {
	var (
		ui                  *testterm.FakeUI
		requirementsFactory *testreq.FakeReqFactory
		configRepo          core_config.ReadWriter
		serviceBrokerRepo   *testapi.FakeServiceBrokerRepo
	)

	BeforeEach(func() {
		configRepo = testconfig.NewRepositoryWithDefaults()

		ui = &testterm.FakeUI{}
		requirementsFactory = &testreq.FakeReqFactory{}
		serviceBrokerRepo = &testapi.FakeServiceBrokerRepo{}
	})

	runCommand := func(args ...string) {
		testcmd.RunCommand(NewRenameServiceBroker(ui, configRepo, serviceBrokerRepo), args, requirementsFactory)
	}

	Describe("requirements", func() {
		It("fails with usage when not invoked with exactly two args", func() {
			requirementsFactory.LoginSuccess = true
			runCommand("welp")
			Expect(ui.FailedWithUsage).To(BeTrue())
		})

		It("fails when not logged in", func() {
			runCommand("okay", "DO---IIIIT")
			Expect(testcmd.CommandDidPassRequirements).To(BeFalse())
		})
	})

	Context("when logged in", func() {
		BeforeEach(func() {
			requirementsFactory.LoginSuccess = true
			broker := models.ServiceBroker{}
			broker.Name = "my-found-broker"
			broker.Guid = "my-found-broker-guid"
			serviceBrokerRepo.FindByNameServiceBroker = broker
		})

		It("renames the given service broker", func() {
			runCommand("my-broker", "my-new-broker")
			Expect(serviceBrokerRepo.FindByNameName).To(Equal("my-broker"))

			Expect(ui.Outputs).To(ContainSubstrings(
				[]string{"Renaming service broker", "my-found-broker", "my-new-broker", "my-user"},
				[]string{"OK"},
			))

			Expect(serviceBrokerRepo.RenamedServiceBrokerGuid).To(Equal("my-found-broker-guid"))
			Expect(serviceBrokerRepo.RenamedServiceBrokerName).To(Equal("my-new-broker"))
		})
	})
})
