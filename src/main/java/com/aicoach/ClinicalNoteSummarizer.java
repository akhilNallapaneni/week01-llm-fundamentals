package com.aicoach;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.anthropic.AnthropicChatModelName;
import dev.langchain4j.model.output.Response;

import java.util.List;

public class ClinicalNoteSummarizer {

    // System prompt: the "role brief" you'd give a junior analyst
    private static final String SYSTEM_PROMPT = """
            You are a clinical documentation specialist.
            Your job is to extract key facts from raw clinical notes for nurse triage intake.
            
            Always respond using EXACTLY this format — no deviations, no extra commentary:
            
            CHIEF_COMPLAINT: <one sentence>
            SYMPTOMS: <comma-separated list>
            DURATION: <e.g. "3 days", "onset today">
            SEVERITY: <LOW | MEDIUM | HIGH>
            FOLLOW_UP_REQUIRED: <YES | NO>
            SUMMARY: <2-3 sentence plain English summary for a non-clinical reader>
            """;

    public static void main(String[] args) {

        // 1. Build the model client — treat this like building a RestTemplate bean
        AnthropicChatModel model = AnthropicChatModel.builder()
                .apiKey(System.getenv("ANTHROPIC_API_KEY"))
                .modelName("claude-haiku-4-5") // fast + cheap for structured tasks
                .temperature(0.0)       // deterministic output — we want consistent format
                .maxTokens(512)         // hard cap — summaries shouldn't need more
                .logRequests(true)      // you'll see the full payload in stdout — study it
                .logResponses(true)
                .build();

        // 2. A realistic clinical note (this is the kind of text that comes in from EHR systems)
        String rawNote = """
                Pt is a 54 y/o male presenting with c/o chest tightness x3 days, \
                worsening with exertion. Reports mild SOB, no fever, no cough. \
                PMHx: HTN, DM2 managed on metformin. Current meds include lisinopril 10mg QD. \
                BP on arrival 148/92. HR 88. SpO2 97% on RA. \
                Patient denies any nausea or diaphoresis. No prior cardiac history documented. \
                Assessment pending cardiology consult.
                """;

        // 3. Compose the conversation: System sets rules, User sends the document
        var response = model.chat(
                List.of(
                        SystemMessage.from(SYSTEM_PROMPT),
                        UserMessage.from("Summarize this clinical note:\n\n" + rawNote)
                )
        );

        // 4. The response is just a String — parse it however you need
        String output = response.aiMessage().text();

        System.out.println("=== STRUCTURED OUTPUT ===");
        System.out.println(output);
        System.out.println("=========================");

        // 5. Demonstrate parsing a field — Week 3 will replace this with proper structured outputs
        String severity = extractField(output, "SEVERITY");
        System.out.println("Extracted severity for routing: " + severity);

        // 6. Token usage — critical for cost control
        System.out.println("\n=== TOKEN USAGE ===");
        System.out.println("Input tokens:  " + response.tokenUsage().inputTokenCount());
        System.out.println("Output tokens: " + response.tokenUsage().outputTokenCount());
        System.out.printf("Estimated cost: ~$%.5f%n",
                estimateCostUSD(response.tokenUsage().inputTokenCount(),
                        response.tokenUsage().outputTokenCount()));
    }

    private static String extractField(String output, String fieldName) {
        for (String line : output.split("\n")) {
            if (line.startsWith(fieldName + ":")) {
                return line.substring(fieldName.length() + 1).trim();
            }
        }
        return "NOT_FOUND";
    }

    // Claude 3.5 Haiku pricing as of 2025: $0.80/M input, $4.00/M output
    private static double estimateCostUSD(int inputTokens, int outputTokens) {
        return (inputTokens / 1_000_000.0 * 0.80) + (outputTokens / 1_000_000.0 * 4.00);
    }
}