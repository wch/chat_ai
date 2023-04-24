from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import time
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import openai
import tiktoken
from htmltools import HTMLDependency
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

import openai_api

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec, TypeGuard
else:
    from typing import ParamSpec, TypeGuard

T = TypeVar("T")
P = ParamSpec("P")

# A place to keep references to Tasks so they don't get GC'd prematurely, as directed in
# asyncio.create_task docs
running_tasks: set[asyncio.Task[Any]] = set()


def safe_create_task(task: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    t = asyncio.create_task(task)
    running_tasks.add(t)
    t.add_done_callback(running_tasks.remove)
    return t


OpenAiModels = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
]

DEFAULT_MODEL: OpenAiModels = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_THROTTLE = 0.1


# A customized version of ChatMessage, with a field for the Markdown `content` converted
# to HTML, and a field for counting the number of tokens in the message.
class ChatMessageEnriched(openai_api.ChatMessage):
    content_preprocessed: str
    content_html: str
    token_count: int


class ChatSession:
    _messages: reactive.Value[tuple[ChatMessageEnriched, ...]]
    _ask_question: Callable[[str, float], None]

    def __init__(
        self,
        messages: reactive.Value[tuple[ChatMessageEnriched, ...]],
        ask_callback: Callable[[str, float], None],
    ):
        self._messages = messages
        self._ask_question = ask_callback

    def messages(self) -> tuple[ChatMessageEnriched, ...]:
        return self._messages()

    def ask(self, query: str, delay: float = 1) -> None:
        self._ask_question(query, delay)


@module.ui
def chat_ui() -> ui.Tag:
    return ui.div(
        {"class": "shiny-gpt-chat"},
        _chat_dependency(),
        ui.output_ui("session_messages_ui"),
        ui.output_ui("current_streaming_message_ui"),
        ui.output_ui("query_ui"),
    )


Renderer = Union[Callable[[str], str], Callable[[str], Awaitable[str]]]


@module.server
def chat_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    *,
    model: OpenAiModels | Callable[[], OpenAiModels] = DEFAULT_MODEL,
    system_prompt: str | Callable[[], str] = DEFAULT_SYSTEM_PROMPT,
    temperature: float | Callable[[], float] = DEFAULT_TEMPERATURE,
    throttle: float | Callable[[], float] = DEFAULT_THROTTLE,
    query_preprocessor: Union[
        Callable[[str], str], Callable[[str], Awaitable[str]]
    ] = lambda x: x,
    input_renderer: Renderer = ui.markdown,
    output_renderer: Renderer = ui.markdown,
    streaming_output_renderer: Renderer = ui.markdown,
) -> ChatSession:
    # Ensure these are functions, even if we were passed static values.
    model = cast(
        # pyright needs a little help with this.
        Callable[[], OpenAiModels],
        wrap_function_nonreactive(model),
    )
    system_prompt = wrap_function_nonreactive(system_prompt)
    temperature = wrap_function_nonreactive(temperature)
    throttle = wrap_function_nonreactive(throttle)

    # If query_preprocessor is not async, wrap it in an async function.
    query_preprocessor = wrap_async(query_preprocessor)
    # Same with input/output renderers.
    input_renderer = wrap_async(input_renderer)
    output_renderer = wrap_async(output_renderer)
    streaming_output_renderer = wrap_async(streaming_output_renderer)

    # This contains a tuple of the most recent messages when a streaming response is
    # coming in. When not streaming, this is set to an empty tuple.
    streaming_chat_messages_batch: reactive.Value[
        tuple[openai_api.ChatCompletionStreaming, ...]
    ] = reactive.Value(tuple())

    # This is the current streaming chat string, in the form of a tuple of strings, one
    # string from each message. When not streaming, it is empty.
    streaming_chat_string_pieces: reactive.Value[tuple[str, ...]] = reactive.Value(
        tuple()
    )

    session_messages: reactive.Value[tuple[ChatMessageEnriched, ...]] = reactive.Value(
        tuple()
    )

    ask_trigger = reactive.Value(0)

    def system_prompt_message() -> ChatMessageEnriched:
        return {
            "role": "system",
            "content_preprocessed": system_prompt(),
            "content": system_prompt(),
            "content_html": "",
            "token_count": get_token_count(system_prompt(), model()),
        }

    @reactive.Effect
    @reactive.event(streaming_chat_messages_batch)
    async def finalize_streaming_result():
        current_batch = streaming_chat_messages_batch()

        for message in current_batch:
            if "content" in message["choices"][0]["delta"]:
                streaming_chat_string_pieces.set(
                    streaming_chat_string_pieces()
                    + (message["choices"][0]["delta"]["content"],)
                )

            if message["choices"][0]["finish_reason"] == "stop":
                # If we got here, we know that streaming_chat_string is not None.
                current_message = "".join(streaming_chat_string_pieces())

                # Update session_messages. We need to make a copy to trigger a reactive
                # invalidation.
                last_message: ChatMessageEnriched = {
                    "content_preprocessed": current_message,
                    "content": current_message,
                    "role": "assistant",
                    "content_html": await output_renderer(current_message),
                    "token_count": get_token_count(current_message, model()),
                }
                session_messages.set(session_messages() + (last_message,))
                streaming_chat_string_pieces.set(tuple())
                return

    @reactive.Effect
    @reactive.event(input.ask, ask_trigger)
    async def perform_query():
        if input.query() == "":
            return

        query_processed = await query_preprocessor(input.query())

        last_message: ChatMessageEnriched = {
            "content_preprocessed": input.query(),
            "content": query_processed,
            "role": "user",
            "content_html": await input_renderer(input.query()),
            "token_count": get_token_count(query_processed, model()),
        }
        session_messages.set(session_messages() + (last_message,))

        # Set this to a non-empty tuple (with a blank string), to indicate that
        # streaming is happening.
        streaming_chat_string_pieces.set(("",))

        # Prepend system prompt, and convert enriched chat messages to chat messages
        # without extra fields.
        session_messages_with_sys_prompt = chat_messages_enriched_to_chat_messages(
            ((system_prompt_message(),) + session_messages())
        )

        # Launch a Task that updates the chat string asynchronously. We run this in a
        # separate task so that the data can come in without need to await it in this
        # Task (which would block other computation to happen, like running reactive
        # stuff).
        messages: StreamResult[openai_api.ChatCompletionStreaming] = stream_to_reactive(
            openai.ChatCompletion.acreate(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
                model=model(),
                messages=session_messages_with_sys_prompt,
                stream=True,
                temperature=temperature(),
            ),
            throttle=throttle(),
        )

        @reactive.Effect
        def copy_messages_to_batch():
            # TODO: messages() can raise an exception; handle it
            streaming_chat_messages_batch.set(messages())

    @output
    @render.ui
    def session_messages_ui():
        messages_html: list[ui.Tag] = []
        for message in session_messages():
            if message["role"] == "system":
                # Don't show system messages.
                continue

            messages_html.append(
                ui.div(
                    {"class": message["role"] + "-message"},
                    message["content_html"],
                )
            )

        return ui.div(*messages_html)

    @output
    @render.ui
    async def current_streaming_message_ui():
        pieces = streaming_chat_string_pieces()

        # Only display this content while streaming. Once the streaming is done, this
        # content will disappear and an identical-looking one will be added to the
        # `session_messages_ui` output.
        if len(pieces) == 0:
            return ui.div()

        content = "".join(pieces)
        if content == "":
            content = "\u2026"  # zero-width string
        else:
            content = await streaming_output_renderer(content)

        return ui.div({"class": "assistant-message"}, content)

    @output
    @render.ui
    @reactive.event(streaming_chat_string_pieces)
    def query_ui():
        # While streaming an answer, don't show the query input.
        if len(streaming_chat_string_pieces()) > 0:
            return ui.div()

        return ui.div(
            ui.input_text_area(
                "query",
                None,
                # value="2+2",
                # placeholder="Ask me anything...",
                rows=1,
                width="100%",
            ),
            ui.div(
                {"style": "width: 100%; text-align: right;"},
                ui.input_action_button("ask", "Ask"),
            ),
            ui.tags.script(
                # The explicit focus() call is needed so that the user can type the next
                # question without clicking on the query box again. However, it's a bit
                # too aggressive, because it will steal focus if, while the answer is
                # streaming, the user clicks somewhere else. It would be better to have
                # the query box set to `display: none` while the answer streams and then
                # unset afterward, so that it can keep focus, but won't steal focus.
                "document.getElementById('%s').focus();"
                % module.resolve_id("query")
            ),
        )

    def ask_question(query: str, delay: float = 1) -> None:
        safe_create_task(delayed_set_query(query, delay))

    async def delayed_set_query(query: str, delay: float) -> None:
        await asyncio.sleep(delay)
        async with reactive._core.lock():
            ui.update_text_area("query", value=query, session=session)
            await reactive.flush()

        # Short delay before triggering ask_trigger.
        safe_create_task(delayed_new_query_trigger(0.2))

    async def delayed_new_query_trigger(delay: float) -> None:
        await asyncio.sleep(delay)
        async with reactive._core.lock():
            ask_trigger.set(ask_trigger() + 1)
            await reactive.flush()

    return ChatSession(session_messages, ask_question)


# ==============================================================================
# Helper functions
# ==============================================================================


def get_token_count(s: str, model: OpenAiModels) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


class StreamResult(Generic[T]):
    _read: Callable[[], tuple[T, ...]]
    _cancel: Callable[[], bool]

    def __init__(self, read: Callable[[], tuple[T, ...]], cancel: Callable[[], bool]):
        self._read = read
        self._cancel = cancel

    def __call__(self) -> tuple[T, ...]:
        """
        Perform a reactive read of the stream. You'll get the latest value, and you will
        receive an invalidation if a new value becomes available.
        """

        return self._read()

    def cancel(self) -> bool:
        """
        Stop the underlying stream from being consumed. Returns False if the task is
        already done or cancelled.
        """
        return self._cancel()


# Converts an async generator of type T into a reactive source of type tuple[T, ...].
def stream_to_reactive(
    func: AsyncGenerator[T, None] | Awaitable[AsyncGenerator[T, None]],
    throttle: float = 0,
) -> StreamResult[T]:
    val: reactive.Value[tuple[T, ...] | Exception] = reactive.Value(tuple())

    async def task_main():
        nonlocal func
        if inspect.isawaitable(func):
            try:
                func = await func  # type: ignore
            except Exception as err:
                val.set(err)
                return
        func = cast(AsyncGenerator[T, None], func)

        last_message_time = time.time()
        message_batch: list[T] = []
        error: Optional[Exception] = None
        try:
            async for message in func:
                message_batch.append(message)

                if time.time() - last_message_time > throttle:
                    async with reactive.lock():
                        val.set(tuple(message_batch))
                        await reactive.flush()

                    last_message_time = time.time()
                    message_batch = []
        except Exception as err:
            error = err

        # Once the stream has ended, flush the remaining messages.
        if len(message_batch) > 0:
            async with reactive.lock():
                val.set(tuple(message_batch))
                await reactive.flush()
        if error:
            val.set(error)
            await reactive.flush()

    task = safe_create_task(task_main())

    def getter() -> tuple[T, ...]:
        result = val.get()
        if isinstance(result, Exception):
            raise result
        else:
            return result

    return StreamResult(getter, lambda: task.cancel())


def chat_message_enriched_to_chat_message(
    msg: ChatMessageEnriched,
) -> openai_api.ChatMessage:
    return {"role": msg["role"], "content": msg["content"]}


def chat_messages_enriched_to_chat_messages(
    messages: Sequence[ChatMessageEnriched],
) -> list[openai_api.ChatMessage]:
    return list(chat_message_enriched_to_chat_message(msg) for msg in messages)


def is_async_callable(
    obj: Callable[P, T] | Callable[P, Awaitable[T]]
) -> TypeGuard[Callable[P, Awaitable[T]]]:
    """
    Returns True if `obj` is an `async def` function, or if it's an object with a
    `__call__` method which is an `async def` function. This function should generally
    be used in this code base instead of iscoroutinefunction().
    """
    if inspect.iscoroutinefunction(obj):
        return True
    if hasattr(obj, "__call__"):  # noqa: B004
        if inspect.iscoroutinefunction(obj.__call__):  # type: ignore
            return True

    return False


def wrap_async(
    fn: Callable[P, T] | Callable[P, Awaitable[T]]
) -> Callable[P, Awaitable[T]]:
    """
    Given a synchronous function that returns T, return an async function that wraps the
    original function. If the input function is already async, then return it unchanged.
    """

    if is_async_callable(fn):
        return fn

    fn = cast(Callable[P, T], fn)

    @functools.wraps(fn)
    async def fn_async(*args: P.args, **kwargs: P.kwargs) -> T:
        return fn(*args, **kwargs)

    return fn_async


def wrap_function_nonreactive(x: T | Callable[[], T]) -> Callable[[], T]:
    """
    This function is used to normalize three types of things so they are wrapped in the
    same kind of object:

    - A value
    - A non-reactive function that return a value
    - A reactive function that returns a value

    All of these will be wrapped in a non-reactive function that returns a value.
    """
    if not callable(x):
        return lambda: x

    @functools.wraps(x)
    def fn_nonreactive() -> T:
        with reactive.isolate():
            return x()

    return fn_nonreactive


def _chat_dependency():
    return HTMLDependency(
        "shiny-gpt-chat",
        "0.0.0",
        source={"package": "chat", "subdir": "assets"},
        script={"src": "chat.js"},
        stylesheet={"href": "chat.css"},
    )
