"""
Service layer + dependency injection patterns from Medical-App-Core.

Patterns covered:
  - Service interface (contracts/)
  - Service implementation (implementations/)
  - DTO ↔ InputDTO separation: InputDTO carries human-readable names,
    service resolves them to UUIDs before calling repository
  - Dependency injection container (initializers/services.go)
  - Error wrapping with fmt.Errorf("%w")
  - Concurrency with errgroup for parallel resolution
  - Third-party service integration (e.g. Stripe)
  - Log service recording duration and status
"""

from __future__ import annotations

from string import Template

GO_VERSIONS = ["1.21", "1.22", "1.23", "1.24"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_SERVICE_CONTRACT_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> contracts

// services/contracts/${entity_lower}ServiceContract.go
// Pattern: service interface decouples controller from implementation
// Observed in: Medical-App-Core/services/contracts/
package contracts

import "github.com/google/uuid"

// ${entity}ServiceContract is the interface every ${entity} service implementation must satisfy.
type ${entity}ServiceContract interface {
\t// CreateFromInput resolves human-readable names in the InputDTO to UUIDs,
\t// then delegates persistence to the repository.
\tCreateFromInput(input ${entity}InputDTO) (uuid.UUID, error)

\tFindBy${lookup_field}(${lookup_param} string) (*${entity}DTO, error)

\tList(page, limit int) ([]${entity}DTO, error)

\tUpdate(id uuid.UUID, dto ${entity}DTO) error

\tDelete(id uuid.UUID) error
}

// ${entity}InputDTO carries fields exactly as received from the HTTP layer.
// Name-based fields (OrganizationName, etc.) are resolved in the service.
type ${entity}InputDTO struct {
\tOrganizationName string `json:"organization_name"`
$input_fields
}

// ${entity}DTO is the canonical data transfer object passed between layers.
type ${entity}DTO struct {
\tOrganizationID string `json:"organization_id"`
$dto_fields
}
</go_file>
""")

_SERVICE_IMPL_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> implementations

// services/implementations/${entity_lower}Service.go
// Pattern: service implementation resolves names→IDs, validates, then calls repo
// Observed in: Medical-App-Core/services/implementations/
package implementations

import (
\t"fmt"

\t"$module/domain/repositories/contracts"
\tsc "$$module/services/contracts"
\t"github.com/google/uuid"
\t"golang.org/x/sync/errgroup"
)

// ${entity}Service implements sc.${entity}ServiceContract.
type ${entity}Service struct {
\trepo         contracts.${entity}Repository
\torgRepo      contracts.OrganizationRepository
}

// New${entity}Service constructs a ${entity}Service with all required repositories.
func New${entity}Service(
\trepo    contracts.${entity}Repository,
\torgRepo contracts.OrganizationRepository,
) *${entity}Service {
\treturn &${entity}Service{repo: repo, orgRepo: orgRepo}
}

// CreateFromInput resolves organization name to UUID in parallel with any other
// lookups, validates the result, then inserts via the repository.
func (s *${entity}Service) CreateFromInput(input sc.${entity}InputDTO) (uuid.UUID, error) {
\tvar (
\t\torgID uuid.UUID
\t)

\t// Parallel resolution of related entities
\tg := &errgroup.Group{}
\tg.Go(func() error {
\t\torg, err := s.orgRepo.FindByName(input.OrganizationName)
\t\tif err != nil {
\t\t\treturn fmt.Errorf("organization %q not found: %w", input.OrganizationName, err)
\t\t}
\t\torgID = org.ID
\t\treturn nil
\t})
\tif err := g.Wait(); err != nil {
\t\treturn uuid.Nil, err
\t}

\tdto := sc.${entity}DTO{
\t\tOrganizationID: orgID.String(),
\t\t// map remaining input fields
\t}
\treturn s.repo.Create(dto)
}

func (s *${entity}Service) FindBy${lookup_field}(${lookup_param} string) (*sc.${entity}DTO, error) {
\tresult, err := s.repo.FindBy${lookup_field}(${lookup_param})
\tif err != nil {
\t\treturn nil, fmt.Errorf("${entity_lower} find by ${lookup_param}: %w", err)
\t}
\treturn result, nil
}

func (s *${entity}Service) List(page, limit int) ([]sc.${entity}DTO, error) {
\treturn s.repo.List(page, limit)
}

func (s *${entity}Service) Update(id uuid.UUID, dto sc.${entity}DTO) error {
\treturn s.repo.Update(id, dto)
}

func (s *${entity}Service) Delete(id uuid.UUID) error {
\treturn s.repo.Delete(id)
}
</go_file>
""")

_DI_CONTAINER_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> initializers

// initializers/services.go
// Pattern: dependency injection container — wires repositories → services
// Observed in: Medical-App-Core/initializers/services.go
package initializers

import (
\t"$module/domain/repositories/contracts"
\timplrepos "$module/infra/repositories"
\tsc "$module/services/contracts"
\tsi "$module/services/implementations"

\t"gorm.io/gorm"
)

// Services is the application-wide DI container.
// It is constructed once in main() and passed to controllers/middleware.
type Services struct {
\tAuthTokenService  sc.AuthTokenServiceContract
\tOrganizationService sc.OrganizationServiceContract
\tUserService       sc.UserServiceContract
\tPatientService    sc.PatientServiceContract
\tDoctorService     sc.DoctorServiceContract
\tAppointmentService sc.AppointmentServiceContract
\tChargeService     sc.ChargeServiceContract
\tMedicalRecordService sc.MedicalRecordServiceContract
}

// InitServices builds and wires all repositories and services.
func InitServices(db *gorm.DB) *Services {
\t// Repositories
\torgRepo         := implrepos.NewOrganizationRepository(db)
\tuserRepo        := implrepos.NewUserRepository(db)
\tpatientRepo     := implrepos.NewPatientRepository(db)
\tdoctorRepo      := implrepos.NewDoctorRepository(db)
\tappointmentRepo := implrepos.NewAppointmentRepository(db)

\t// Services (inject repos as interface values)
\tauthSvc  := si.NewAuthTokenService()
\torgSvc   := si.NewOrganizationService(orgRepo)
\tuserSvc  := si.NewUserService(userRepo, orgRepo)
\tpatSvc   := si.NewPatientService(patientRepo, orgRepo)
\tdocSvc   := si.NewDoctorService(doctorRepo, orgRepo)
\tapptSvc  := si.NewAppointmentService(
\t\tappointmentRepo, orgRepo, patientRepo, userRepo, doctorRepo,
\t)
\tchargeSvc := si.NewChargeService(orgRepo)

\treturn &Services{
\t\tAuthTokenService:      authSvc,
\t\tOrganizationService:   orgSvc,
\t\tUserService:           userSvc,
\t\tPatientService:        patSvc,
\t\tDoctorService:         docSvc,
\t\tAppointmentService:    apptSvc,
\t\tChargeService:         chargeSvc,
\t}
}
</go_file>
""")

_RABBITMQ_CONSUMER_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> events

// services/events/doctors/doctorEventConsumer.go
// Pattern: RabbitMQ consumer with reconnect loop + JSON unmarshaling
// Observed in: Medical-App-Core/services/events/
package events

import (
\t"encoding/json"
\t"fmt"
\t"log"
\t"os"
\t"time"

\t"github.com/streadway/amqp"
)

// DoctorCreatedEvent is the message payload published when a doctor is registered.
type DoctorCreatedEvent struct {
\tDoctorID   string    `json:"doctor_id"`
\tFullName   string    `json:"full_name"`
\tSpecialty  string    `json:"specialty"`
\tCreatedAt  time.Time `json:"created_at"`
}

// DoctorEventConsumer subscribes to the doctors.created queue.
type DoctorEventConsumer struct {
\tconn    *amqp.Connection
\tch      *amqp.Channel
\tqueue   string
\thandler func(DoctorCreatedEvent) error
}

// NewDoctorEventConsumer dials RabbitMQ and declares the queue.
func NewDoctorEventConsumer(handler func(DoctorCreatedEvent) error) (*DoctorEventConsumer, error) {
\turl := os.Getenv("RABBITMQ_BASE_URL")
\tif url == "" {
\t\treturn nil, fmt.Errorf("RABBITMQ_BASE_URL is not set")
\t}

\tconn, err := amqp.Dial(url)
\tif err != nil {
\t\treturn nil, fmt.Errorf("rabbitmq dial: %w", err)
\t}

\tch, err := conn.Channel()
\tif err != nil {
\t\tconn.Close()
\t\treturn nil, fmt.Errorf("rabbitmq channel: %w", err)
\t}

\tqueue := "doctors.created"
\t_, err = ch.QueueDeclare(queue, true, false, false, false, nil)
\tif err != nil {
\t\tch.Close(); conn.Close()
\t\treturn nil, fmt.Errorf("queue declare: %w", err)
\t}

\treturn &DoctorEventConsumer{conn: conn, ch: ch, queue: queue, handler: handler}, nil
}

// Consume starts consuming messages and blocks until ctx is cancelled.
func (c *DoctorEventConsumer) Consume() error {
\tmsgs, err := c.ch.Consume(c.queue, "", false, false, false, false, nil)
\tif err != nil {
\t\treturn fmt.Errorf("consume: %w", err)
\t}

\tlog.Printf("consuming from %s", c.queue)
\tfor msg := range msgs {
\t\tvar event DoctorCreatedEvent
\t\tif err := json.Unmarshal(msg.Body, &event); err != nil {
\t\t\tlog.Printf("unmarshal error: %v — nacking", err)
\t\t\tmsg.Nack(false, false) // dead-letter
\t\t\tcontinue
\t\t}
\t\tif err := c.handler(event); err != nil {
\t\t\tlog.Printf("handler error: %v — nacking with requeue", err)
\t\t\tmsg.Nack(false, true)
\t\t\tcontinue
\t\t}
\t\tmsg.Ack(false)
\t}
\treturn nil
}

// Close releases RabbitMQ resources.
func (c *DoctorEventConsumer) Close() {
\tc.ch.Close()
\tc.conn.Close()
}
</go_file>
""")

ENTITIES = [
    dict(
        entity="Patient",
        entity_lower="patient",
        lookup_field="CPF",
        lookup_param="cpf",
        input_fields='\tName    string `json:"name"`\n\tContact string `json:"contact"`\n\tCPF     string `json:"cpf,omitempty"`',
        dto_fields='\tName    string `json:"name"`\n\tContact string `json:"contact"`\n\tCPF     string `json:"cpf,omitempty"`',
    ),
    dict(
        entity="Doctor",
        entity_lower="doctor",
        lookup_field="CRM",
        lookup_param="crm",
        input_fields='\tFullName   string `json:"full_name"`\n\tCPF        string `json:"cpf"`\n\tCRM        string `json:"crm"`\n\tSpecialty  string `json:"specialty"`',
        dto_fields='\tFullName   string `json:"full_name"`\n\tCRM        string `json:"crm"`\n\tSpecialty  string `json:"specialty"`',
    ),
    dict(
        entity="Appointment",
        entity_lower="appointment",
        lookup_field="Status",
        lookup_param="status",
        input_fields='\tPatientName    string    `json:"patient_name"`\n\tDoctorFullName string    `json:"doctor_full_name"`\n\tSpecialization string    `json:"specialization"`\n\tDateTime       time.Time `json:"date_time"`',
        dto_fields='\tPatientID      string    `json:"patient_id"`\n\tDoctorID       string    `json:"doctor_id"`\n\tSpecialization string    `json:"specialization"`\n\tDateTime       time.Time `json:"date_time"`\n\tStatus         string    `json:"status"`',
    ),
]


class ServicePatternGenerator:
    """Generate service layer + DI training examples."""

    MODULE = "medical-sas-api"

    def all_examples(self) -> list[str]:
        examples: list[str] = []
        for ent in ENTITIES:
            for ver in GO_VERSIONS:
                examples.append(
                    _SERVICE_CONTRACT_TEMPLATE.substitute(
                        go_version=ver, module=self.MODULE, **ent
                    )
                )
                examples.append(
                    _SERVICE_IMPL_TEMPLATE.substitute(
                        go_version=ver, module=self.MODULE, **ent
                    )
                )
        for ver in GO_VERSIONS:
            examples.append(_DI_CONTAINER_TEMPLATE.substitute(go_version=ver, module=self.MODULE))
            examples.append(_RABBITMQ_CONSUMER_TEMPLATE.substitute(go_version=ver))
        return examples
