"""
Test patterns extracted from Medical-App-Core.

Patterns covered:
  - go-sqlmock + GORM mock DB setup helper
  - testify assert / require style
  - Success + error path table-driven tests
  - Fiber app.Test() for HTTP handler testing
  - Mock repository pattern (interface-based)
  - setupTestDB() helper function
  - ExpectBegin / ExpectQuery / ExpectCommit / ExpectRollback chain
"""

from __future__ import annotations

from string import Template

GO_VERSIONS = ["1.21", "1.22", "1.23", "1.24"]

_SERVICE_TEST_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> implementations_test

// services/implementations/${entity_lower}_service_test.go
// Pattern: sqlmock + GORM + testify — success and error paths
// Observed in: Medical-App-Core/services/implementations/appointment_service_test.go
package implementations_test

import (
\t"errors"
\t"testing"
\t"time"

\t"$module/services/contracts"
\t"$module/services/implementations"

\t"github.com/DATA-DOG/go-sqlmock"
\t"github.com/google/uuid"
\t"github.com/stretchr/testify/assert"
\t"gorm.io/driver/postgres"
\t"gorm.io/gorm"
)

// setupTestDB creates an in-memory mock database for service tests.
// Returns the GORM DB and the sqlmock controller for expectation setup.
func setupTestDB(t *testing.T) (*gorm.DB, sqlmock.Sqlmock) {
\tt.Helper()
\tdb, mock, err := sqlmock.New()
\tif err != nil {
\t\tt.Fatalf("failed to create mock DB: %v", err)
\t}
\tgormDB, err := gorm.Open(postgres.New(postgres.Config{Conn: db}), &gorm.Config{})
\tif err != nil {
\t\tt.Fatalf("failed to open GORM DB: %v", err)
\t}
\tt.Cleanup(func() { db.Close() })
\treturn gormDB, mock
}

// TestCreate_Success verifies that a valid DTO produces a non-nil UUID.
func Test${entity}Service_Create_Success(t *testing.T) {
\tdb, mock := setupTestDB(t)
\tsvc := implementations.New${entity}Service(db)

\tdto := contracts.${entity}DTO{
$dto_fields
\t}

\tnewID := uuid.New()
\tmock.ExpectBegin()
\tmock.ExpectQuery(`INSERT INTO "${table}" .* RETURNING "id"`).
\t\tWithArgs(
\t\t\tsqlmock.AnyArg(), // created_at
\t\t\tsqlmock.AnyArg(), // updated_at
\t\t\tnil,              // deleted_at
$mock_args
\t\t\tsqlmock.AnyArg(), // id
\t\t).
\t\tWillReturnRows(sqlmock.NewRows([]string{"id"}).AddRow(newID))
\tmock.ExpectCommit()

\tgotID, err := svc.Create(dto)

\tassert.NoError(t, err)
\tassert.Equal(t, newID, gotID)
\tassert.NoError(t, mock.ExpectationsWereMet())
}

// Test${entity}Service_Create_DBError verifies the service surfaces DB errors.
func Test${entity}Service_Create_DBError(t *testing.T) {
\tdb, mock := setupTestDB(t)
\tsvc := implementations.New${entity}Service(db)

\tdto := contracts.${entity}DTO{
$dto_fields
\t}

\tmock.ExpectBegin()
\tmock.ExpectQuery(`INSERT INTO "${table}" .* RETURNING "id"`).
\t\tWillReturnError(errors.New("connection refused"))
\tmock.ExpectRollback()

\tgotID, err := svc.Create(dto)

\tassert.Error(t, err)
\tassert.Equal(t, uuid.Nil, gotID)
\tassert.NoError(t, mock.ExpectationsWereMet())
}

// Test${entity}Service_FindBy${lookup_field}_NotFound verifies gorm.ErrRecordNotFound is surfaced.
func Test${entity}Service_FindBy${lookup_field}_NotFound(t *testing.T) {
\tdb, mock := setupTestDB(t)
\tsvc := implementations.New${entity}Service(db)

\tmock.ExpectQuery(`SELECT .* FROM "${table}"`).
\t\tWithArgs("NOTEXIST", sqlmock.AnyArg()).
\t\tWillReturnError(gorm.ErrRecordNotFound)

\tresult, err := svc.FindBy${lookup_field}("NOTEXIST")

\tassert.Error(t, err)
\tassert.Nil(t, result)
\tassert.NoError(t, mock.ExpectationsWereMet())
}
</go_file>
""")

_HTTP_HANDLER_TEST_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> controllers_test

// controllers/${entity_lower}_controller_test.go
// Pattern: Fiber app.Test() for HTTP handler unit testing
// Observed in: Medical-App-Core/utils/response_util_test.go
package controllers_test

import (
\t"bytes"
\t"encoding/json"
\t"net/http"
\t"net/http/httptest"
\t"testing"

\t"$module/controllers"
\t"github.com/gofiber/fiber/v2"
\t"github.com/stretchr/testify/require"
)

// mock${entity}Service implements contracts.${entity}ServiceContract for testing.
type mock${entity}Service struct {
\tcreateResult string
\tcreateErr    error
}

func (m *mock${entity}Service) CreateFromInput(input any) (any, error) {
\treturn m.createResult, m.createErr
}

func Test${entity}Controller_Create_Success(t *testing.T) {
\tapp := fiber.New()

\tctrl := controllers.New${entity}Controller(&mock${entity}Service{
\t\tcreateResult: "test-uuid",
\t})
\tapp.Post("/${entity_plural}", ctrl.Create${entity})

\tbody, _ := json.Marshal(map[string]any{
\t\t"organization_name": "Test Org",
$request_fields
\t})

\treq := httptest.NewRequest(http.MethodPost, "/${entity_plural}", bytes.NewReader(body))
\treq.Header.Set("Content-Type", "application/json")

\tresp, err := app.Test(req, -1)
\trequire.NoError(t, err)
\trequire.Equal(t, http.StatusCreated, resp.StatusCode)

\tvar result map[string]any
\trequire.NoError(t, json.NewDecoder(resp.Body).Decode(&result))
\trequire.Equal(t, "${entity} created successfully", result["message"])
}

func Test${entity}Controller_Create_MissingField(t *testing.T) {
\tapp := fiber.New()
\tctrl := controllers.New${entity}Controller(&mock${entity}Service{})
\tapp.Post("/${entity_plural}", ctrl.Create${entity})

\t// Send empty body — required fields missing
\treq := httptest.NewRequest(http.MethodPost, "/${entity_plural}", bytes.NewReader([]byte("{}")))
\treq.Header.Set("Content-Type", "application/json")

\tresp, err := app.Test(req, -1)
\trequire.NoError(t, err)
\trequire.Equal(t, http.StatusBadRequest, resp.StatusCode)
}

func Test${entity}Controller_Create_InvalidJSON(t *testing.T) {
\tapp := fiber.New()
\tctrl := controllers.New${entity}Controller(&mock${entity}Service{})
\tapp.Post("/${entity_plural}", ctrl.Create${entity})

\treq := httptest.NewRequest(http.MethodPost, "/${entity_plural}", bytes.NewReader([]byte("not json")))
\treq.Header.Set("Content-Type", "application/json")

\tresp, err := app.Test(req, -1)
\trequire.NoError(t, err)
\trequire.Equal(t, http.StatusBadRequest, resp.StatusCode)
}
</go_file>
""")

_TABLE_DRIVEN_TEST_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> utils_test

// utils/response_util_test.go
// Pattern: table-driven tests with Fiber app.Test()
// Observed in: Medical-App-Core/utils/response_util_test.go
package utils_test

import (
\t"encoding/json"
\t"net/http"
\t"net/http/httptest"
\t"testing"

\t"$module/utils"
\t"github.com/gofiber/fiber/v2"
\t"github.com/stretchr/testify/require"
)

func TestRespondWithJSON(t *testing.T) {
\tru := &utils.JSONResponseUtil{}

\ttests := []struct {
\t\tname       string
\t\tcode       int
\t\tpayload    any
\t\twantStatus int
\t\twantKey    string
\t}{
\t\t{
\t\t\tname:       "200 ok with map payload",
\t\t\tcode:       http.StatusOK,
\t\t\tpayload:    map[string]string{"message": "ok"},
\t\t\twantStatus: http.StatusOK,
\t\t\twantKey:    "message",
\t\t},
\t\t{
\t\t\tname:       "201 created",
\t\t\tcode:       http.StatusCreated,
\t\t\tpayload:    map[string]string{"id": "abc-123"},
\t\t\twantStatus: http.StatusCreated,
\t\t\twantKey:    "id",
\t\t},
\t\t{
\t\t\tname:       "400 bad request",
\t\t\tcode:       http.StatusBadRequest,
\t\t\tpayload:    map[string]string{"error": "invalid input"},
\t\t\twantStatus: http.StatusBadRequest,
\t\t\twantKey:    "error",
\t\t},
\t}

\tfor _, tc := range tests {
\t\tt.Run(tc.name, func(t *testing.T) {
\t\t\tapp := fiber.New()
\t\t\tapp.Get("/test", func(c *fiber.Ctx) error {
\t\t\t\treturn ru.RespondWithJSON(c, tc.code, tc.payload)
\t\t\t})

\t\t\treq := httptest.NewRequest(http.MethodGet, "/test", nil)
\t\t\tresp, err := app.Test(req, -1)
\t\t\trequire.NoError(t, err)
\t\t\trequire.Equal(t, tc.wantStatus, resp.StatusCode)
\t\t\trequire.Equal(t, "application/json", resp.Header.Get("Content-Type"))

\t\t\tvar body map[string]any
\t\t\trequire.NoError(t, json.NewDecoder(resp.Body).Decode(&body))
\t\t\t_, ok := body[tc.wantKey]
\t\t\trequire.True(t, ok, "expected key %q in response", tc.wantKey)
\t\t})
\t}
}
</go_file>
""")

ENTITIES = [
    dict(
        entity="Appointment",
        entity_lower="appointment",
        entity_plural="appointments",
        table="appointments",
        lookup_field="Status",
        dto_fields='\t\tOrganizationID: uuid.New(),\n\t\tPatientID:      uuid.New(),\n\t\tDoctorID:       uuid.New(),\n\t\tDateTime:       time.Now(),\n\t\tStatus:         "scheduled",',
        mock_args='\t\t\tsqlmock.AnyArg(), // organization_id\n\t\t\tsqlmock.AnyArg(), // patient_id\n\t\t\tsqlmock.AnyArg(), // doctor_id\n\t\t\tsqlmock.AnyArg(), // date_time\n\t\t\t"scheduled",      // status',
        request_fields='\t\t"patient_name":    "João Silva",\n\t\t"doctor_full_name": "Dr. Maria",\n\t\t"specialization":  "Cardiology",\n\t\t"date_time":       "2024-03-01T10:00:00Z",',
    ),
    dict(
        entity="Patient",
        entity_lower="patient",
        entity_plural="patients",
        table="patients",
        lookup_field="CPF",
        dto_fields='\t\tOrganizationID: uuid.New().String(),\n\t\tName:           "João Silva",\n\t\tContact:        "+55 11 99999-9999",',
        mock_args='\t\t\tsqlmock.AnyArg(), // organization_id\n\t\t\t"João Silva",     // name\n\t\t\t"+55 11 99999-9999", // contact',
        request_fields='\t\t"name":    "João Silva",\n\t\t"contact": "+55 11 99999-9999",',
    ),
]


class TestPatternGenerator:
    """Generate test pattern training examples."""

    MODULE = "medical-sas-api"

    def all_examples(self) -> list[str]:
        examples: list[str] = []
        for ent in ENTITIES:
            for ver in GO_VERSIONS:
                examples.append(
                    _SERVICE_TEST_TEMPLATE.substitute(go_version=ver, module=self.MODULE, **ent)
                )
                examples.append(
                    _HTTP_HANDLER_TEST_TEMPLATE.substitute(go_version=ver, module=self.MODULE, **ent)
                )
        for ver in GO_VERSIONS:
            examples.append(_TABLE_DRIVEN_TEST_TEMPLATE.substitute(go_version=ver, module=self.MODULE))
        return examples
