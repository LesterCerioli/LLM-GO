
from __future__ import annotations

from string import Template

ENTITIES = [
    dict(
        entity="Patient",
        table="patients",
        entity_lower="patient",
        entity_plural="patients",
        fields='''\tOrganizationID uuid.UUID      `gorm:"type:uuid;not null;index" json:"organization_id"`
\tName           string         `gorm:"type:varchar(255);not null" json:"name" validate:"required"`
\tContact        string         `gorm:"type:varchar(50);not null" json:"contact" validate:"required"`
\tCPF            string         `gorm:"type:varchar(14);uniqueIndex" json:"cpf,omitempty"`
\tSSN            string         `gorm:"type:varchar(11);uniqueIndex" json:"ssn,omitempty"`
\tDOB            time.Time      `gorm:"type:date" json:"dob"`
\tGender         string         `gorm:"type:varchar(20)" json:"gender,omitempty"`
\tAddress        string         `gorm:"type:text" json:"address,omitempty"`''',
        lookup_field="CPF",
        lookup_column="cpf",
    ),
    dict(
        entity="Doctor",
        table="doctors",
        entity_lower="doctor",
        entity_plural="doctors",
        fields='''\tOrganizationID uuid.UUID `gorm:"type:uuid;not null;index" json:"organization_id"`
\tFullName       string    `gorm:"type:varchar(255);not null" json:"full_name" validate:"required"`
\tCPF            string    `gorm:"type:varchar(14);uniqueIndex" json:"cpf"`
\tCRM            string    `gorm:"type:varchar(20);uniqueIndex" json:"crm"`
\tSpecialty      string    `gorm:"type:varchar(100);not null" json:"specialty" validate:"required"`
\tEmail          string    `gorm:"type:varchar(255);uniqueIndex" json:"email" validate:"required,email"`
\tPhone          string    `gorm:"type:varchar(20)" json:"phone,omitempty"`''',
        lookup_field="CRM",
        lookup_column="crm",
    ),
    dict(
        entity="Appointment",
        table="appointments",
        entity_lower="appointment",
        entity_plural="appointments",
        fields='''\tOrganizationID  uuid.UUID  `gorm:"type:uuid;not null;index" json:"organization_id"`
\tPatientID       uuid.UUID  `gorm:"type:uuid;not null;index" json:"patient_id"`
\tDoctorID        uuid.UUID  `gorm:"type:uuid;not null;index" json:"doctor_id"`
\tUserID          uuid.UUID  `gorm:"type:uuid;not null" json:"user_id"`
\tSpecialization  string     `gorm:"type:varchar(100);not null" json:"specialization"`
\tDateTime        time.Time  `gorm:"type:timestamptz;not null" json:"date_time"`
\tStatus          string     `gorm:"type:varchar(50);default:scheduled" json:"status"`
\tNotes           string     `gorm:"type:text" json:"notes,omitempty"`''',
        lookup_field="Status",
        lookup_column="status",
    ),
    dict(
        entity="User",
        table="users",
        entity_lower="user",
        entity_plural="users",
        fields='''\tOrganizationID uuid.UUID `gorm:"type:uuid;not null;index" json:"organization_id"`
\tName           string    `gorm:"type:varchar(255);not null" json:"name" validate:"required"`
\tEmail          string    `gorm:"type:varchar(255);uniqueIndex;not null" json:"email" validate:"required,email"`
\tPassword       string    `gorm:"type:varchar(255);not null" json:"-"` // never serialised
\tRole           string    `gorm:"type:varchar(50);not null;default:user" json:"role" validate:"required,oneof=admin user doctor nurse"`''',
        lookup_field="Email",
        lookup_column="email",
    ),
]

GO_VERSIONS = ["1.21", "1.22", "1.23", "1.24"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_ENTITY_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> entities

// domain/entities/${entity_lower}.go
// Pattern: GORM entity with UUID PK + soft delete + JSON/validate tags
// Observed in: Medical-App-Core/domain/entities/
package entities

import (
\t"time"

\t"github.com/google/uuid"
\t"gorm.io/gorm"
)

// ${entity} represents the ${table} database table.
type ${entity} struct {
\tID        uuid.UUID      `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
\tCreatedAt time.Time      `gorm:"autoCreateTime" json:"created_at"`
\tUpdatedAt time.Time      `gorm:"autoUpdateTime" json:"updated_at"`
\tDeletedAt gorm.DeletedAt `gorm:"index" json:"-"` // soft delete
$fields
}

// TableName overrides GORM's convention-based table name.
func (${entity}) TableName() string { return "${table}" }

// BeforeCreate sets a UUID before the record is inserted.
func (e *${entity}) BeforeCreate(tx *gorm.DB) error {
\tif e.ID == uuid.Nil {
\t\te.ID = uuid.New()
\t}
\treturn nil
}
</go_file>
""")

_REPO_INTERFACE_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> contracts

// domain/repositories/contracts/${entity_lower}Repository.go
// Pattern: repository interface (contract) — domain layer knows nothing about GORM
// Observed in: Medical-App-Core/domain/repositories/contracts/
package contracts

import (
\t"$module/domain/dtos"
\t"github.com/google/uuid"
)

// ${entity}Repository defines the persistence contract for ${entity} entities.
// Implementations live in infra/repositories/ and depend on GORM or any other ORM.
type ${entity}Repository interface {
\tCreate(dto dtos.${entity}DTO) (uuid.UUID, error)
\tFindByID(id uuid.UUID) (*dtos.${entity}DTO, error)
\tFindBy${lookup_field}(${lookup_column} string) (*dtos.${entity}DTO, error)
\tList(page, limit int) ([]dtos.${entity}DTO, error)
\tUpdate(id uuid.UUID, dto dtos.${entity}DTO) error
\tDelete(id uuid.UUID) error // soft delete
}
</go_file>
""")

_REPO_IMPL_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> repositories

// infra/repositories/${entity_lower}Repository.go
// Pattern: GORM implementation of the repository interface
// Observed in: Medical-App-Core/infra/repositories/
package repositories

import (
\t"fmt"

\t"$module/domain/dtos"
\t"$module/domain/entities"
\t"github.com/google/uuid"
\t"gorm.io/gorm"
)

// ${entity}RepositoryImpl implements contracts.${entity}Repository using GORM.
type ${entity}RepositoryImpl struct {
\tdb *gorm.DB
}

// New${entity}Repository constructs a new ${entity}RepositoryImpl.
func New${entity}Repository(db *gorm.DB) *${entity}RepositoryImpl {
\treturn &${entity}RepositoryImpl{db: db}
}

func (r *${entity}RepositoryImpl) Create(dto dtos.${entity}DTO) (uuid.UUID, error) {
\tentity := entities.${entity}{} // map DTO → entity fields
\t// (field mapping omitted for brevity — use a mapper or manual assignment)
\tif err := r.db.Create(&entity).Error; err != nil {
\t\treturn uuid.Nil, fmt.Errorf("${entity_lower} create: %w", err)
\t}
\treturn entity.ID, nil
}

func (r *${entity}RepositoryImpl) FindByID(id uuid.UUID) (*dtos.${entity}DTO, error) {
\tvar entity entities.${entity}
\tif err := r.db.First(&entity, "id = ?", id).Error; err != nil {
\t\tif err == gorm.ErrRecordNotFound {
\t\t\treturn nil, fmt.Errorf("${entity_lower} not found: %s", id)
\t\t}
\t\treturn nil, fmt.Errorf("${entity_lower} find: %w", err)
\t}
\tdto := dtos.${entity}DTO{} // map entity → DTO
\treturn &dto, nil
}

func (r *${entity}RepositoryImpl) FindBy${lookup_field}(${lookup_column} string) (*dtos.${entity}DTO, error) {
\tvar entity entities.${entity}
\tif err := r.db.Where("${lookup_column} = ?", ${lookup_column}).First(&entity).Error; err != nil {
\t\tif err == gorm.ErrRecordNotFound {
\t\t\treturn nil, fmt.Errorf("${entity_lower} with ${lookup_column}=%s not found", ${lookup_column})
\t\t}
\t\treturn nil, fmt.Errorf("${entity_lower} find by ${lookup_column}: %w", err)
\t}
\tdto := dtos.${entity}DTO{}
\treturn &dto, nil
}

func (r *${entity}RepositoryImpl) List(page, limit int) ([]dtos.${entity}DTO, error) {
\tvar entities []entities.${entity}
\toffset := (page - 1) * limit
\tif err := r.db.Offset(offset).Limit(limit).Find(&entities).Error; err != nil {
\t\treturn nil, fmt.Errorf("${entity_lower} list: %w", err)
\t}
\tdtos := make([]dtos.${entity}DTO, 0, len(entities))
\t// map each entity to DTO
\treturn dtos, nil
}

func (r *${entity}RepositoryImpl) Update(id uuid.UUID, dto dtos.${entity}DTO) error {
\tresult := r.db.Model(&entities.${entity}{}).Where("id = ?", id).Updates(dto)
\tif result.Error != nil {
\t\treturn fmt.Errorf("${entity_lower} update: %w", result.Error)
\t}
\tif result.RowsAffected == 0 {
\t\treturn fmt.Errorf("${entity_lower} not found: %s", id)
\t}
\treturn nil
}

func (r *${entity}RepositoryImpl) Delete(id uuid.UUID) error {
\t// GORM soft delete: sets deleted_at instead of removing the row
\tresult := r.db.Delete(&entities.${entity}{}, "id = ?", id)
\tif result.Error != nil {
\t\treturn fmt.Errorf("${entity_lower} delete: %w", result.Error)
\t}
\tif result.RowsAffected == 0 {
\t\treturn fmt.Errorf("${entity_lower} not found: %s", id)
\t}
\treturn nil
}
</go_file>
""")

_DB_INIT_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> initializers

// initializers/database.go
// Pattern: GORM + PostgreSQL connection with pool config + env-based DSN
// Observed in: Medical-App-Core/initializers/database.go
package initializers

import (
\t"fmt"
\t"log"
\t"os"
\t"strings"
\t"time"

\t"gorm.io/driver/postgres"
\t"gorm.io/gorm"
\t"gorm.io/gorm/logger"
)

// InitialDB opens a PostgreSQL connection using environment variables.
// Panics on failure — the app cannot run without a database.
func InitialDB() *gorm.DB {
\tdsn := buildDSN()

\tdb, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
\t\tLogger: logger.Default.LogMode(logger.Info),
\t})
\tif err != nil {
\t\tlog.Fatalf("database connection failed: %v", err)
\t}

\tsqlDB, err := db.DB()
\tif err != nil {
\t\tlog.Fatalf("failed to get underlying sql.DB: %v", err)
\t}
\tsqlDB.SetMaxOpenConns(25)
\tsqlDB.SetMaxIdleConns(10)
\tsqlDB.SetConnMaxLifetime(5 * time.Minute)

\tlog.Println("database connected:", maskDSN(dsn))
\treturn db
}

func buildDSN() string {
\treturn fmt.Sprintf(
\t\t"host=%s port=%s user=%s password=%s dbname=%s sslmode=%s TimeZone=%s",
\t\tgetEnv("DB_HOST",     "localhost"),
\t\tgetEnv("DB_PORT",     "5432"),
\t\tgetEnv("DB_USER",     "postgres"),
\t\tos.Getenv("DB_PASSWORD"),
\t\tgetEnv("DB_NAME",     "appdb"),
\t\tgetEnv("DB_SSLMODE",  "disable"),
\t\tgetEnv("DB_TIMEZONE", "UTC"),
\t)
}

func getEnv(key, fallback string) string {
\tif v := os.Getenv(key); v != "" {
\t\treturn v
\t}
\treturn fallback
}

func maskDSN(dsn string) string {
\tpassword := os.Getenv("DB_PASSWORD")
\tif password == "" {
\t\treturn dsn
\t}
\treturn strings.ReplaceAll(dsn, password, "*****")
}
</go_file>
""")


class GormPatternGenerator:
    """Generate GORM + PostgreSQL training examples."""

    def all_examples(self) -> list[str]:
        examples: list[str] = []
        module = "medical-sas-api"
        for ent in ENTITIES:
            for ver in GO_VERSIONS:
                examples.append(_ENTITY_TEMPLATE.substitute(go_version=ver, module=module, **ent))
                examples.append(_REPO_INTERFACE_TEMPLATE.substitute(go_version=ver, module=module, **ent))
                examples.append(_REPO_IMPL_TEMPLATE.substitute(go_version=ver, module=module, **ent))
        for ver in GO_VERSIONS:
            examples.append(_DB_INIT_TEMPLATE.substitute(go_version=ver))
        return examples
