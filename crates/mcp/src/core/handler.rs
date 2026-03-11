//! SMG client handler for MCP server notifications and elicitation.
//!
//! Implements RMCP's `ClientHandler` trait to handle:
//! - Elicitation requests (approval flow)
//! - Tool/resource/prompt list change notifications
//! - Progress and logging notifications

use std::{future::Future, sync::Arc};

use parking_lot::RwLock;
use rmcp::{
    model::{
        CancelledNotificationParam, ClientInfo, CreateElicitationRequestParam,
        CreateElicitationResult, LoggingLevel, LoggingMessageNotificationParam,
        ProgressNotificationParam, ResourceUpdatedNotificationParam,
    },
    service::{NotificationContext, RequestContext},
    ClientHandler, RoleClient,
};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};

use crate::{
    approval::{ApprovalManager, ApprovalMode, ApprovalOutcome, ApprovalParams},
    inventory::ToolInventory,
    tenant::TenantContext,
};

/// Request to refresh server inventory.
#[derive(Debug, Clone)]
pub struct RefreshRequest {
    pub server_key: String,
}

/// Per-request context set before tool execution, cleared after.
#[derive(Debug, Clone)]
pub struct HandlerRequestContext {
    pub request_id: String,
    pub approval_mode: ApprovalMode,
    pub tenant_ctx: TenantContext,
}

impl HandlerRequestContext {
    pub fn new(
        request_id: impl Into<String>,
        approval_mode: ApprovalMode,
        tenant_ctx: TenantContext,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            approval_mode,
            tenant_ctx,
        }
    }
}

#[derive(Clone)]
pub struct SmgClientHandler {
    server_key: Arc<str>,
    approval_manager: Arc<ApprovalManager>,
    #[expect(dead_code)]
    tool_inventory: Arc<ToolInventory>,
    client_info: ClientInfo,
    request_ctx: Arc<RwLock<Option<HandlerRequestContext>>>,
    request_ctx_lock: Arc<Mutex<()>>,
    refresh_tx: Option<mpsc::Sender<RefreshRequest>>,
}

impl SmgClientHandler {
    pub fn new(
        server_key: impl AsRef<str>,
        approval_manager: Arc<ApprovalManager>,
        tool_inventory: Arc<ToolInventory>,
    ) -> Self {
        let mut client_info = ClientInfo::default();
        client_info.client_info.name = "smg".to_string();
        client_info.client_info.version = env!("CARGO_PKG_VERSION").to_string();

        Self {
            server_key: Arc::from(server_key.as_ref()),
            approval_manager,
            tool_inventory,
            client_info,
            request_ctx: Arc::new(RwLock::new(None)),
            request_ctx_lock: Arc::new(Mutex::new(())),
            refresh_tx: None,
        }
    }

    #[must_use]
    pub fn with_refresh_channel(mut self, tx: mpsc::Sender<RefreshRequest>) -> Self {
        self.refresh_tx = Some(tx);
        self
    }

    #[must_use]
    pub fn with_client_info(mut self, info: ClientInfo) -> Self {
        self.client_info = info;
        self
    }

    pub fn set_request_context(&self, ctx: HandlerRequestContext) {
        *self.request_ctx.write() = Some(ctx);
    }

    pub fn clear_request_context(&self) {
        *self.request_ctx.write() = None;
    }

    pub fn request_context(&self) -> Option<HandlerRequestContext> {
        self.request_ctx.read().clone()
    }

    pub async fn run_with_request_context<F, Fut, T>(&self, ctx: HandlerRequestContext, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        struct RequestContextResetGuard<'a>(&'a SmgClientHandler);

        impl Drop for RequestContextResetGuard<'_> {
            fn drop(&mut self) {
                self.0.clear_request_context();
            }
        }

        let _guard = self.request_ctx_lock.lock().await;
        self.set_request_context(ctx);
        let _reset_guard = RequestContextResetGuard(self);
        f().await
    }

    pub fn server_key(&self) -> &str {
        &self.server_key
    }

    fn send_refresh(&self) {
        if let Some(tx) = &self.refresh_tx {
            let _ = tx
                .try_send(RefreshRequest {
                    server_key: self.server_key.to_string(),
                })
                .map_err(|e| {
                    warn!(
                        server_key = %self.server_key,
                        error = %e,
                        "Failed to send refresh request"
                    );
                });
        }
    }
}

impl ClientHandler for SmgClientHandler {
    async fn create_elicitation(
        &self,
        request: CreateElicitationRequestParam,
        context: RequestContext<RoleClient>,
    ) -> Result<CreateElicitationResult, rmcp::ErrorData> {
        use crate::annotations::ToolAnnotations;

        let elicitation_id = match &context.id {
            rmcp::model::RequestId::String(s) => s.to_string(),
            rmcp::model::RequestId::Number(n) => n.to_string(),
        };

        // Get request context
        let req_ctx = self.request_ctx.read().clone().ok_or_else(|| {
            rmcp::ErrorData::internal_error("No request context set for elicitation", None)
        })?;

        // Use message as the tool identifier (elicitation doesn't have tool name directly)
        let message = &request.message;

        // Default annotations (conservative - not read-only, potentially destructive)
        let hints = ToolAnnotations::default();

        let params = ApprovalParams {
            request_id: &req_ctx.request_id,
            server_key: &self.server_key,
            elicitation_id: &elicitation_id,
            tool_name: "elicitation",
            hints: &hints,
            message,
            tenant_ctx: &req_ctx.tenant_ctx,
        };

        let outcome = self
            .approval_manager
            .handle_approval(req_ctx.approval_mode, params)
            .await
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        match outcome {
            ApprovalOutcome::Decided(decision) => {
                if decision.is_allowed() {
                    Ok(CreateElicitationResult {
                        action: rmcp::model::ElicitationAction::Accept,
                        content: None,
                    })
                } else {
                    Ok(CreateElicitationResult {
                        action: rmcp::model::ElicitationAction::Decline,
                        content: None,
                    })
                }
            }
            ApprovalOutcome::Pending { rx, .. } => {
                // Wait for user response
                match rx.await {
                    Ok(decision) => {
                        if decision.is_approved() {
                            Ok(CreateElicitationResult {
                                action: rmcp::model::ElicitationAction::Accept,
                                content: None,
                            })
                        } else {
                            Ok(CreateElicitationResult {
                                action: rmcp::model::ElicitationAction::Decline,
                                content: None,
                            })
                        }
                    }
                    Err(_) => Err(rmcp::ErrorData::internal_error(
                        "Approval channel closed",
                        None,
                    )),
                }
            }
        }
    }

    async fn on_cancelled(
        &self,
        params: CancelledNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        info!(
            server_key = %self.server_key,
            request_id = %params.request_id,
            reason = ?params.reason,
            "MCP server cancelled request"
        );
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        debug!(
            server_key = %self.server_key,
            token = ?params.progress_token,
            progress = %params.progress,
            total = ?params.total,
            message = ?params.message,
            "MCP server progress"
        );
    }

    async fn on_resource_updated(
        &self,
        params: ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        info!(
            server_key = %self.server_key,
            uri = %params.uri,
            "MCP server resource updated"
        );
    }

    async fn on_resource_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!(server_key = %self.server_key, "MCP server resource list changed");
        self.send_refresh();
    }

    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!(server_key = %self.server_key, "MCP server tool list changed");
        self.send_refresh();
    }

    async fn on_prompt_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!(server_key = %self.server_key, "MCP server prompt list changed");
        self.send_refresh();
    }

    fn get_info(&self) -> ClientInfo {
        self.client_info.clone()
    }

    async fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        let logger = params.logger.as_deref().unwrap_or("mcp");

        match params.level {
            LoggingLevel::Emergency | LoggingLevel::Alert | LoggingLevel::Critical => {
                error!(
                    server_key = %self.server_key,
                    logger = %logger,
                    level = ?params.level,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Error => {
                error!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Warning => {
                warn!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Notice | LoggingLevel::Info => {
                info!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
            LoggingLevel::Debug => {
                debug!(
                    server_key = %self.server_key,
                    logger = %logger,
                    "MCP: {}",
                    params.data
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::{audit::AuditLog, policy::PolicyEngine};

    fn test_handler() -> SmgClientHandler {
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(audit_log.clone()));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));
        let tool_inventory = Arc::new(ToolInventory::new());

        SmgClientHandler::new("test-server", approval_manager, tool_inventory)
    }

    #[test]
    fn test_handler_creation() {
        let handler = test_handler();
        assert_eq!(handler.server_key(), "test-server");
        assert!(handler.request_context().is_none());
    }

    #[test]
    fn test_request_context() {
        let handler = test_handler();

        let ctx = HandlerRequestContext::new(
            "req-1",
            ApprovalMode::PolicyOnly,
            TenantContext::new("tenant-1"),
        );

        handler.set_request_context(ctx.clone());
        assert!(handler.request_context().is_some());

        let retrieved = handler.request_context().unwrap();
        assert_eq!(retrieved.request_id, "req-1");

        handler.clear_request_context();
        assert!(handler.request_context().is_none());
    }

    #[tokio::test]
    async fn test_run_with_request_context_scopes_and_clears() {
        let handler = test_handler();
        let ctx = HandlerRequestContext::new(
            "req-1",
            ApprovalMode::PolicyOnly,
            TenantContext::new("tenant-1"),
        );

        let handler_clone = handler.clone();
        handler
            .run_with_request_context(ctx.clone(), move || async move {
                let current = handler_clone.request_context();
                assert!(current.is_some());
                assert_eq!(current.unwrap().request_id, "req-1");
            })
            .await;

        assert!(handler.request_context().is_none());
    }

    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "test verifies panic unwinding cleanup behavior in spawned task"
    )]
    async fn test_run_with_request_context_clears_on_panic() {
        use std::sync::Arc;

        let handler = Arc::new(test_handler());
        let panic_handler = Arc::clone(&handler);
        let panic_task = tokio::spawn(async move {
            panic_handler
                .run_with_request_context(
                    HandlerRequestContext::new(
                        "req-panic",
                        ApprovalMode::PolicyOnly,
                        TenantContext::new("tenant-panic"),
                    ),
                    || async {
                        panic!("intentional panic to verify context cleanup");
                    },
                )
                .await;
        });

        assert!(panic_task.await.is_err());
        assert!(handler.request_context().is_none());
    }

    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "test concurrency: spawned tasks are awaited before test exits"
    )]
    async fn test_run_with_request_context_serializes_concurrent_calls() {
        use std::sync::Arc;

        use tokio::sync::oneshot;

        let handler = Arc::new(test_handler());
        let (started_tx, started_rx) = oneshot::channel::<()>();
        let (release_tx, release_rx) = oneshot::channel::<()>();

        let first_handler = Arc::clone(&handler);
        let first_task = tokio::spawn(async move {
            let first_ctx = HandlerRequestContext::new(
                "req-1",
                ApprovalMode::PolicyOnly,
                TenantContext::new("tenant-1"),
            );
            first_handler
                .run_with_request_context(first_ctx, move || async move {
                    let _ = started_tx.send(());
                    let _ = release_rx.await;
                })
                .await;
        });

        let _ = started_rx.await;
        assert_eq!(
            handler.request_context().map(|ctx| ctx.request_id),
            Some("req-1".to_string())
        );

        let second_handler = Arc::clone(&handler);
        let second_task = tokio::spawn(async move {
            let second_ctx = HandlerRequestContext::new(
                "req-2",
                ApprovalMode::PolicyOnly,
                TenantContext::new("tenant-2"),
            );
            let second_handler_clone = Arc::clone(&second_handler);
            second_handler
                .run_with_request_context(second_ctx, move || async move {
                    second_handler_clone
                        .request_context()
                        .map(|ctx| ctx.request_id)
                })
                .await
        });

        tokio::task::yield_now().await;
        assert_eq!(
            handler.request_context().map(|ctx| ctx.request_id),
            Some("req-1".to_string())
        );

        let _ = release_tx.send(());
        first_task.await.unwrap();

        let second_context_id = second_task.await.unwrap();
        assert_eq!(second_context_id.as_deref(), Some("req-2"));
        assert!(handler.request_context().is_none());
    }

    #[test]
    fn test_client_info() {
        let handler = test_handler();
        let info = handler.get_info();
        assert_eq!(info.client_info.name, "smg");
    }

    #[test]
    fn test_with_refresh_channel() {
        let (tx, _rx) = mpsc::channel(10);
        let handler = test_handler().with_refresh_channel(tx);
        assert!(handler.refresh_tx.is_some());
    }
}
